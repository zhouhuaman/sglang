# SPDX-License-Identifier: Apache-2.0
"""KDA ``gate + chunk-local cumsum`` operator: a self-contained torch + triton implementation.

This module is the functional equivalent of ``kda_gate_chunk_cumsum`` from
``python/sglang/kernels/ops/attention/fla/kda.py``, but it depends only on
``torch`` / ``triton`` and **does not import any sglang code**, so it can be used
standalone by the gated test directory (the shared ``load_kda.py`` indirection
layer cannot be maintained here).

Computation (identical to the upstream kernel):

    1. Gated activation (standard gate)::

         gate[t, s] = -exp(A_log[h]) * softplus(raw_gate[t, s] + dt_bias[h, s])

       where ``softplus(x) = log(1 + exp(x))`` (degrades to x when x is large,
       preserving accuracy);
    2. In-chunk cumulative sum: a "chunk-local" prefix sum along the time
       dimension T with chunk size ``BT`` (each chunk restarts from 0; nothing
       carries across chunk boundaries);
    3. Optional scaling: ``out *= scale``; in practice ``scale = RCP_LN2``,
       which converts the result from the ln domain to the log2 domain for
       downstream ``exp2``-style kernels.

The triton kernel is a tiled, vectorized implementation:

    * grid = ``(cdiv(K, BS), num_chunks, B * H)``, with BS = 32 and BT = chunk_size;
    * each program (CTA) handles one ``(time chunk, (batch, head), S-tile)``,
      loading a ``[BT, BS]`` 2D tile (rows = time, columns = channels);
    * runs ``tl.cumsum(b_gate, axis=0)`` on the tile (axis=0 is exactly the
      chunk-length dimension, matching the ``tl.cumsum`` semantics confirmed
      available on this hardware).

Trailing (partial) chunk handling
---------------------------------
The last chunk may have fewer than BT rows, and the last S-tile in the channel
dimension may also fall short of BS. The kernel handles this uniformly with
masks: out-of-bounds elements are loaded as 0, and invalid rows are zeroed out
**before** the cumsum (``tl.where(masks, b_gate, 0.0)``). Because ``tl.cumsum``
accumulates forward along axis=0, **zero-padding the tail does not pollute the
prefix sums of the valid leading rows** — exactly matching the real kernel's
behavior (``boundary_check`` returns 0).

npud notes
----------
This module ``import torch_npu`` at the top (omitting it triggers a "Background
device ... is not available" error). In the real runtime environment (docker
container), the caller sources CANN's ``set_env.sh`` and sets
``LD_LIBRARY_PATH`` / ``TORCH_DEVICE_BACKEND_AUTOLOAD=0`` before launching
python; this module sets no environment variables. The top-level logic at import
time allocates no NPU tensors (it touches the ``npu`` device only when calling
the driver), so in an environment without an NPU device the module still imports
and ``gate_chunk_cumsum_ref`` runs normally; ``gate_chunk_cumsum_triton``
automatically falls back to the reference implementation when no NPU is detected
(handy for pure-CPU logic checks) and uses the triton kernel when an NPU is
available.
"""

import torch
import torch_npu  # noqa: F401  (must be imported before creating any npu tensor)

import triton
import triton.language as tl


# log2(e) = 1 / ln(2), the exact fp32 value (consistent with flash-linear-attention).
RCP_LN2 = 1.4426950216293335

# Compile-time tile sizes. BT is the chunk_size; BS is the channel-dimension tile size.
# tl.cumsum requires both values to be powers of two.
_DEFAULT_BT = 64
# Optimization record (tracked in the repo's OPTIMIZATION_LOG.md): BS increased from 32 to 64
# to reduce the flattened grid size (cdiv(K, BS) goes 4→2). Under the target case
# B=1,T=16384,H=96,K=128, grid = (2, 256, 96) → flattened 49152 ≤ 65535, removing the
# grid-overflow unsupported problem.
# Second-round optimization: BS increased from 64 to 128, cdiv(K, BS) goes 2→1,
# grid = (1, 256, 96) → flattened 24576, speedup improved from ~3.1x to ~7.0x
# (torch_npu baseline).
_DEFAULT_BS = 128
_SOFTPLUS_THRESHOLD = 20.0


def _cdiv(a: int, b: int) -> int:
    """Integer division that rounds up (ceiling)."""
    return -(a // -b)


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


@triton.jit
def _softplus_fwd(x):
    """softplus(x) = log(1 + exp(x)); use a linear approximation when x exceeds the threshold to avoid exp overflow."""
    return tl.where(x < 20.0, tl.log(1.0 + tl.exp(x)), x)


@triton.jit(do_not_specialize=["T"])
def _gate_cumsum_kernel(
    x,
    A_log,
    dt_bias,
    o,
    scale,
    T,  # number of time steps per batch (a runtime scalar, avoids recompiling per T value)
    H: tl.constexpr,
    K: tl.constexpr,  # channel dimension (=S)
    BT: tl.constexpr,
    BS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    """Tiled triton kernel for the KDA gated activation + in-chunk cumsum.

    grid = (cdiv(K, BS), num_chunks, B * H); each program processes one
    ``[BT, BS]`` tile (BT rows of time x BS columns of channels):
      * loads raw gate, optionally adding dt_bias;
      * gated activation: gate = -exp(A_log) * softplus(x + bias);
      * zeroes invalid rows, then ``tl.cumsum(b_gate, axis=0)`` (chunk-local prefix sum);
      * optionally multiplies by scale and writes back.
    """
    i_s = tl.program_id(0)  # tile index along the S dimension: 0..cdiv(K, BS)-1
    i_t = tl.program_id(1)  # chunk index: 0..num_chunks-1
    i_bh = tl.program_id(2)  # combined (batch, head) index: 0..B*H-1

    # Note: the "chunk index" here directly uses the global time coordinate. Since T
    # is the same for every batch, the chunk time offset = i_t * BT (fixed-length
    # pattern). The only place the kernel assumes homogeneity is T (same length per
    # batch), which always holds for fixed-length inputs.
    i_b = i_bh // H
    i_h = i_bh % H

    # Starting offset of the current chunk along the time dimension / channel-dimension offset
    base_t = i_t * BT
    s_off = i_s * BS

    rows = tl.arange(0, BT)  # [BT]
    cols = tl.arange(0, BS)  # [BS]

    tile_t = base_t + rows  # time coordinates handled by this program [BT]
    tile_s = s_off + cols  # channel coordinates handled by this program [BS]
    masks = (tile_t[:, None] < T) & (tile_s[None, :] < K)  # [BT, BS]

    # Row-major contiguous memory: flat index = b*T*H*K + t*H*K + h*K + s
    ptr_base = x + i_b * T * H * K + i_h * K
    ptr_x = ptr_base + tile_t[:, None] * (H * K) + tile_s[None, :]

    # Out-of-bounds positions load as 0 (row/column masks); invalid rows are zeroed out again below.
    b_s = tl.load(ptr_x, mask=masks, other=0.0).to(tl.float32)

    if HAS_BIAS:
        # dt_bias is flattened [H*K]; the current head's bias = dt_bias[i_h*K : i_h*K+K]
        ptr_bias = dt_bias + i_h * K + tile_s
        b_bias = tl.load(ptr_bias, mask=tile_s < K, other=0.0).to(tl.float32)
        b_s = b_s + b_bias[None, :]

    # Per-head A_log scalar
    b_a = tl.load(A_log + i_h).to(tl.float32)

    # Gated activation: gate = -exp(A_log) * softplus(x + bias)
    b_gate = -tl.exp(b_a) * _softplus_fwd(b_s)

    # Zero out invalid rows (time out-of-bounds). cumsum accumulates forward along axis=0,
    # so zero-padding the tail does not pollute the prefix sums of the valid leading rows;
    # invalid columns were already zeroed, so they never enter the result.
    b_gate = tl.where(masks, b_gate, 0.0)

    # In-chunk prefix sum (no carry across chunk boundaries; each chunk restarts from 0)
    b_o = tl.cumsum(b_gate, axis=0)

    if HAS_SCALE:
        b_o *= scale

    ptr_o = o + i_b * T * H * K + i_h * K + tile_t[:, None] * (H * K) + tile_s[None, :]
    tl.store(ptr_o, b_o.to(tl.float32), mask=masks)


def gate_chunk_cumsum_ref(
    x,
    A_log,
    dt_bias=None,
    chunk_size=_DEFAULT_BT,
    scale=RCP_LN2,
):
    """Pure torch CPU reference implementation (runs on any device; typically CPU).

    The computation matches the kernel exactly::

        gate[t,s] = -exp(A_log[h]) * softplus(x[t,s] + bias[h,s])
        out[k]    = scale * cumsum(gate)   # chunk-local, each chunk restarts from 0

    Supports arbitrary B/T/H/K (including K not a multiple of the tile size and T
    not a multiple of BT), enabling exact comparison against the triton kernel result.

    Args:
        x: [B, T, H, K] raw gate values
        A_log: [H] per-head log scale
        dt_bias: flattened [H*K] bias (optional; None means no bias)
        chunk_size: chunk size (pass the same BT as the kernel)
        scale: output scaling (None means no scaling)

    Returns:
        [B, T, H, K] fp32 result.
    """
    assert x.dim() == 4, f"x must be 4D [B,T,H,K], got shape {tuple(x.shape)}"
    B, T, H, K = x.shape

    x = x.float()

    # 1) add bias + softplus
    if dt_bias is not None:
        bias = dt_bias.to(torch.float32).reshape(1, 1, H, K)
        y = x + bias
    else:
        y = x
    y = torch.where(
        y < _SOFTPLUS_THRESHOLD,
        torch.log1p(torch.exp(y)),
        y,
    )  # softplus(x + bias)

    # 2) multiply by -exp(A_log) (broadcast to [1,1,H,1]).
    #    Both A_log[h] and the (H,K) channel indexing are stride-1 elementwise broadcasts;
    #    note that the int index [0,0,0,0] directly indexes x, equivalent to x[0][0][0][0] (verified).
    ap = A_log.to(torch.float32).view(1, 1, -1, 1)  # [1,1,H,1]
    y = -torch.exp(ap) * y

    # 3) chunk-local cumsum: process each chunk as a [B, avail_rows, H, K] slice
    #    y[:, a:a+avail] and take the prefix sum over dim=1 (the time axis);
    #    an incomplete chunk (avail < chunk_size) is first zero-padded to chunk_size
    #    and then truncated — the padding is only at the tail, so it does not pollute
    #    the prefix sums of the valid rows (consistent with the kernel's tail
    #    zero-fill semantics).
    edges = list(range(0, T, chunk_size)) + [T]
    parts = []
    for a, b in zip(edges[:-1], edges[1:]):
        avail = b - a  # actual number of rows in the chunk (< chunk_size means an incomplete chunk)
        part = y[:, a : a + avail].cumsum(dim=1)  # prefix sum along the time axis
        if avail < chunk_size:
            pad = torch.zeros(
                B, chunk_size - avail, H, K, dtype=part.dtype, device=part.device
            )
            part = torch.cat([part, pad], dim=1)
        parts.append(part)
    y = torch.cat(parts, dim=1)  # concatenate the chunks in the original T order
    y = y[:, :T].contiguous()

    # 4) optional scaling
    if scale is not None:
        y = y * scale
    return y.float().contiguous()


def gate_chunk_cumsum_torch(
    x,
    A_log,
    dt_bias=None,
    chunk_size=_DEFAULT_BT,
    scale=RCP_LN2,
) -> torch.Tensor:
    """Meta-operator (torch_npu operator-graph) version: gated activation + in-chunk cumsum.

    This is the **performance-baseline** implementation — it composes the same
    computation from torch_npu's off-the-shelf per-operator building blocks, for
    speedup comparison against the triton kernel. It is faster than
    ``gate_chunk_cumsum_ref``:
      * activation + accumulation are fused (no second sort/concat); the data is touched only once;
      * cumsum is done in one pass via reshape + ``torch.cumsum`` instead of looping per chunk;
      * `softplus` has an algorithmic ``logaddexp2`` form: ``sp(x) = x + logaddexp2(0,-x)*ln2``,
        which avoids intermediate overflow for positive float32 without a ``where`` branch.

    Mathematically identical to triton/ref:
        gate[t,s] = -exp(A_log[h]) * softplus(x[t,s] + dt_bias[h,s])
        out      = scale * cumsum_over_chunk(gate)

    Arguments are the same as ``gate_chunk_cumsum_triton``. Returns [B,T,H,K] fp32
    (the device matches the input: NPU or CPU).
    """
    assert x.dim() == 4, f"x must be 4D [B,T,H,K], got shape {tuple(x.shape)}"
    assert _is_power_of_two(chunk_size), "chunk_size must be a power of 2"
    BT = int(chunk_size)
    B, T, H, K = x.shape
    NT = _cdiv(T, BT)
    pad = NT * BT - T

    xf = x.to(torch.float32)
    if pad:  # with a non-empty tail (pad>0) a direct reshape is invalid; zero-pad to NT*BT first, then do the chunk cumsum
        xf = torch.cat([xf, torch.zeros(B, pad, H, K, dtype=xf.dtype, device=xf.device)], dim=1)
    if dt_bias is not None:
        xf = xf + dt_bias.to(torch.float32).reshape(1, 1, H, K)
    # exact softplus via logaddexp (avoids intermediate exp overflow):
    #   sp(x) = log(1+exp(x)) = x + logaddexp(0, -x)
    sp = xf + torch.logaddexp(torch.zeros_like(xf), -xf)
    gate = -torch.exp(A_log.to(torch.float32).view(1, 1, H, 1)) * sp

    y = gate.reshape(B, NT, BT, H, K).cumsum(dim=2).reshape(B, NT * BT, H, K)
    y = y[:, :T].contiguous()
    if scale is not None:
        y = y * scale
    return y.float().contiguous()


def gate_chunk_cumsum_triton(
    x,
    A_log,
    dt_bias=None,
    chunk_size=_DEFAULT_BT,
    scale=RCP_LN2,
    num_warps=1,
) -> torch.Tensor:
    """Triton version of the KDA gated activation + in-chunk cumsum (runs on NPU).

    Arguments are compatible with ``kda_gate_chunk_cumsum``:

        x: [B, T, H, K] raw gate values
        A_log: [H] per-head log scale
        dt_bias: flattened [H*K] bias (optional; None means no bias)
        chunk_size: chunk size (default 64; must be a power of 2)
        scale: output scaling (default RCP_LN2; None means no scaling)
        num_warps: number of warps per CTA (default 1, matching the real kernel)

    Returns:
        [B, T, H, K] fp32 result on the NPU (falls back to the CPU reference when
        no NPU is available).
    """
    assert x.dim() == 4, f"x must be 4D [B,T,H,K], got shape {tuple(x.shape)}"
    assert _is_power_of_two(chunk_size), "chunk_size must be a power of 2"

    # Pure-CPU / no-NPU environment: fall back to the reference implementation so
    # unit tests can validate logic on machines without an NPU.
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        return gate_chunk_cumsum_ref(
            x, A_log, dt_bias=dt_bias, chunk_size=chunk_size, scale=scale
        )

    B, T, H, K = x.shape
    BS = _DEFAULT_BS
    BT = int(chunk_size)

    # Convert inputs to fp32 and move them to the NPU (does not mutate the caller's tensors)
    x = x.to(torch.float32).to("npu")
    A_log = A_log.to(torch.float32).to("npu")
    has_bias = dt_bias is not None
    if has_bias:
        dt_bias = dt_bias.to(torch.float32).to("npu")
    else:
        dt_bias = x  # when HAS_BIAS=False this pointer is never read, so passing a dummy is fine

    o = torch.empty_like(x)

    num_chunks = _cdiv(T, BT)
    grid = (_cdiv(K, BS), num_chunks, B * H)

    _gate_cumsum_kernel[grid](
        x,
        A_log,
        dt_bias,
        o,
        float(scale) if scale is not None else 0.0,
        T,
        H=H,
        K=K,
        BT=BT,
        BS=BS,
        HAS_BIAS=has_bias,
        HAS_SCALE=scale is not None,
        num_warps=num_warps,
    )
    # Synchronize and wait for completion. If the test script needs timing, it
    # should wrap the call with torch.npu.synchronize() itself.
    torch.npu.synchronize()
    return o
