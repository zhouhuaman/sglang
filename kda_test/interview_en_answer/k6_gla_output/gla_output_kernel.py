# SPDX-License-Identifier: Apache-2.0
"""Standalone implementation of the KDA ``chunk_gla_fwd_o_gk`` operator (Kernel 6 / GLA Output).

This module is the functional equivalent of ``chunk_gla_fwd_o_gk`` +
``chunk_gla_fwd_kernel_o`` in ``python/sglang/kernels/ops/attention/fla/kda.py``,
but depends only on ``torch`` / ``triton`` -- it does **not import any sglang code**,
so it can be driven standalone by the tests in this directory.

What it computes (identical to the upstream kernel)::

    o[t] = o_cross[t] + o_intra[t]

  * Cross-chunk (cross):
        q_gated[t, k] = q[t, k] * scale * exp2(g[t, k])
        o_cross[t, v] = sum_k q_gated[t, k] * h[chunk(t), v, k]
                      = (q_gated @ h^T)[t, v]
  * Intra-chunk (intra):
        A_masked = where(lower_triangular, Aqk, 0)
        o_intra[t, v] = sum_j A_masked[t, j] * v_new[j, v]
                      = (A_masked @ v_new)[t, v]

The triton kernel is a tiled blocked-matmul implementation:

    * grid = ``(cdiv(V, BV), NT, B * H)``, BK=32, BV=32, BT=chunk_size;
    * each program (CTA) handles one ``(V-tile, chunk, (batch, head))`` intersection
      and loads the ``[BT, BV]`` output tile;
    * the K dimension is a sequential loop: each iteration loads the ``[BT, BK]``
      q/g tiles and the ``[BV, BK]`` h tile, and accumulates ``tl.dot(q_gated, h^T)``
      into ``b_o``;
    * the intra-chunk part loads the ``[BT, BT]`` Aqk tile (lower-triangular mask
      applied) and the ``[BT, BV]`` v_new tile, and accumulates ``tl.dot(A_masked,
      v_new)`` into ``b_o``.

Trailing (partial) chunk handling
---------------------------------
The last chunk may have fewer rows than BT, and the last tile along the V dim may
also be shorter than BV. The kernel handles all of this uniformly with
``boundary_check``: out-of-bounds elements are loaded as 0, and once
``tl.where(m_s, ...)`` applies the causal mask, the ``b_A`` of out-of-bounds rows is
also zeroed, so it cannot pollute the accumulator.

Environment notes
-----------------
This module does ``import torch_npu`` at the top (without it you trigger the
"Background device ... is not available" error). For the real runtime, the caller
sources CANN's ``set_env.sh`` before launching python and sets ``LD_LIBRARY_PATH``
and ``TORCH_DEVICE_BACKEND_AUTOLOAD=0``. In an environment with no NPU device the
module can still be imported and ``gla_output_ref`` run normally;
``gla_output_triton`` automatically falls back to the reference implementation when
no NPU is detected (for pure-CPU logic checking).
"""

import os

import torch
import torch_npu  # noqa: F401  (must be imported before creating any npu tensor)

import triton
import triton.language as tl

# log2(e) = 1 / ln(2), the exact fp32 value (matching flash-linear-attention).
RCP_LN2 = 1.4426950216293335

# Compile-time tile sizes. BT is chunk_size; BK is the K-dim tile, BV the V-dim tile.
_DEFAULT_BT = 64
_DEFAULT_BK = 32
_DEFAULT_BV = 128

# Iteration experiment parameters (env override; defaults match the converged config)
# NW=4 was optimal in the 2026-08-25 experiment (7.38ms->6.93ms, -6%); NS had no effect.
_K6_NW = int(os.getenv("K6_NW", "4"))
_K6_NS = int(os.getenv("K6_NS", "1"))


def _cdiv(a: int, b: int) -> int:
    """Integer division rounding up."""
    return -(a // -b)


# ---------------------------------------------------------------------------
# CPU reference implementation (ground truth, runs on any device)
# ---------------------------------------------------------------------------


def gla_output_ref(
    q,
    v_new,
    g,
    Aqk,
    h,
    scale,
    chunk_size=_DEFAULT_BT,
):
    """Pure-torch CPU reference implementation (runs on any device, typically CPU).

    Computes exactly what the kernel computes::

        o_cross[t] = (q[t] * exp2(g[t]) * scale) @ h[chunk(t)]^T
        o_intra[t] = (Aqk[t] * tril) @ v_new[t]
        o[t]        = o_cross[t] + o_intra[t]

    Supports arbitrary B/T/H/K/V (including a trailing chunk shorter than BT and
    K/V not multiples of the tile sizes), so the triton kernel output can be
    compared precisely.

    Args:
        q:     [B, T, H, K]     query vectors (bf16/fp16/fp32 all fine; internally cast to fp32)
        v_new: [B, T, H, V]    corrected values (Kernel 5 output)
        g:     [B, T, H, K]    cumulative gate (Kernel 1 output, log2 space)
        Aqk:   [B, T, H, BT]   intra-chunk causal attention weights
        h:     [B, NT, H, V, K] compressed-state snapshots (Kernel 5 output)
        scale: float           attention scale factor 1/sqrt(K)
        chunk_size: int        chunk size (should match upstream; default 64)

    Returns:
        [B, T, H, V] fp32 result.
    """
    assert q.dim() == 4, f"q must be 4D [B,T,H,K], got shape {tuple(q.shape)}"
    assert v_new.dim() == 4, f"v_new must be 4D, got shape {tuple(v_new.shape)}"
    assert h.dim() == 5, f"h must be 5D [B,NT,H,V,K], got shape {tuple(h.shape)}"

    B, T, H, K = q.shape
    V = v_new.shape[-1]
    BT = int(chunk_size)
    NT = _cdiv(T, BT)

    qf = q.float()
    vf = v_new.float()
    gf = g.float()
    Af = Aqk.float()
    hf = h.float()

    o = torch.zeros(B, T, H, V, dtype=torch.float32)

    for b in range(B):
        for c in range(NT):
            tc = c * BT
            tc_end = min(T, tc + BT)
            BT_act = tc_end - tc
            for h_idx in range(H):
                q_chunk = qf[b, tc:tc_end, h_idx]         # [BT_act, K]
                g_chunk = gf[b, tc:tc_end, h_idx]         # [BT_act, K]
                v_chunk = vf[b, tc:tc_end, h_idx]         # [BT_act, V]
                A_chunk = Af[b, tc:tc_end, h_idx, :BT_act]  # [BT_act, BT_act]
                h_s = hf[b, c, h_idx]                     # [V, K]

                qg = q_chunk * torch.exp2(g_chunk) * scale  # [BT_act, K]
                o_cross = qg @ h_s.T                        # [BT_act, V]

                causal_mask = torch.tril(
                    torch.ones(BT_act, BT_act, dtype=torch.float32)
                )
                o_intra = (A_chunk * causal_mask) @ v_chunk  # [BT_act, V]

                o[b, tc:tc_end, h_idx] = o_cross + o_intra
    return o.contiguous()


# ---------------------------------------------------------------------------
# torch_npu meta-operator version (performance baseline)
# ---------------------------------------------------------------------------


def gla_output_torch(
    q,
    v_new,
    g,
    Aqk,
    h,
    scale,
    chunk_size=_DEFAULT_BT,
):
    """Meta-operator (torch_npu operator-graph) version: mathematically identical to the triton kernel.

    This is the **performance baseline** implementation -- it does the same
    computation as a composition of torch_npu's ready-made per-operator ops, to be
    compared against the triton kernel for speedup. Unlike ``gla_output_ref``'s
    per-chunk loop, this implementation vectorizes the chunk/head dimensions fully
    and does it all with batched ``matmul`` calls in one pass:

      * ``q * exp2(g) * scale`` elementwise;
      * ``o_cross = matmul(q_gated, h.transpose(-1,-2))``  batched [BT,K]@[K,V];
      * ``A_masked = A * tril``, then ``matmul(A_masked, v_new)``  batched [BT,BT]@[BT,V];
      * add the two, reshape back to [B, T, H, V].

    Parameters are the same as ``gla_output_triton``. Returns [B, T, H, V] fp32
    (device matches the inputs).
    """
    assert q.dim() == 4, f"q must be 4D [B,T,H,K], got shape {tuple(q.shape)}"
    assert h.dim() == 5, f"h must be 5D [B,NT,H,V,K], got shape {tuple(h.shape)}"

    B, T, H, K = q.shape
    V = v_new.shape[-1]
    BT = int(chunk_size)
    NT = _cdiv(T, BT)

    # Zero-pad the tail up to NT*BT so every chunk has BT rows (zero rows do not pollute the result).
    pad = NT * BT - T
    if pad:
        q = torch.cat(
            [q, torch.zeros(B, pad, H, K, dtype=q.dtype, device=q.device)], dim=1
        )
        v_new = torch.cat(
            [v_new, torch.zeros(B, pad, H, V, dtype=v_new.dtype, device=v_new.device)],
            dim=1,
        )
        g = torch.cat(
            [g, torch.zeros(B, pad, H, K, dtype=g.dtype, device=g.device)], dim=1
        )
        Aqk = torch.cat(
            [
                Aqk,
                torch.zeros(B, pad, H, BT, dtype=Aqk.dtype, device=Aqk.device),
            ],
            dim=1,
        )

    # ── reshape to [B, NT, BT, H, ...] ──
    q_r = q.reshape(B, NT, BT, H, K).float()
    v_r = v_new.reshape(B, NT, BT, H, V).float()
    g_r = g.reshape(B, NT, BT, H, K).float()
    A_r = Aqk.reshape(B, NT, BT, H, BT).float()
    h_r = h.float()  # [B, NT, H, V, K]

    # ── cross-chunk: o_cross = (q * exp2(g) * scale) @ h^T ──
    qg = q_r * torch.exp2(g_r) * scale  # [B, NT, BT, H, K]
    h_t = h_r.transpose(-1, -2)         # [B, NT, H, K, V]
    # Move the H dim into matmul's batch dim: [B, NT, H, BT, K] @ [B, NT, H, K, V]
    qg_p = qg.permute(0, 1, 3, 2, 4)    # [B, NT, H, BT, K]
    o_cross = torch.matmul(qg_p, h_t)   # [B, NT, H, BT, V]
    o_cross = o_cross.permute(0, 1, 3, 2, 4)  # [B, NT, BT, H, V]

    # ── intra-chunk: o_intra = (A * tril) @ v_new ──
    mask = torch.tril(
        torch.ones(BT, BT, dtype=torch.float32, device=q.device)
    )  # [BT, BT]
    # A_r: [B, NT, BT, H, BT] — mask is [BT, BT], so it must line up with the last dim (BT)
    A_masked = A_r * mask.view(1, 1, BT, 1, BT)
    A_p = A_masked.permute(0, 1, 3, 2, 4)  # [B, NT, H, BT, BT]
    v_p = v_r.permute(0, 1, 3, 2, 4)       # [B, NT, H, BT, V]
    o_intra = torch.matmul(A_p, v_p)       # [B, NT, H, BT, V]
    o_intra = o_intra.permute(0, 1, 3, 2, 4)  # [B, NT, BT, H, V]

    o = (o_cross + o_intra).reshape(B, NT * BT, H, V)
    o = o[:, :T].contiguous()
    return o.float().contiguous()


# ---------------------------------------------------------------------------
# Triton kernel (unit under test)
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["T"])
def chunk_gla_fwd_kernel_o(
    q,
    v,
    g,
    h,
    o,
    A,
    scale,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)  # 24

    NT = T // BT
    v_tiles = V // BV

    total = NT * H * v_tiles
    per_core = tl.cdiv(total, num_programs)

    s_q_t: tl.constexpr = H * K
    s_v_t: tl.constexpr = H * V
    s_h_v: tl.constexpr = K
    s_a_t: tl.constexpr = H * BT

    r = tl.arange(0, BT)
    c = tl.arange(0, BT)

    rk = tl.arange(0, BK)
    rv = tl.arange(0, BV)

    m_s = (
        r[:, None].to(tl.float32)
        >= c[None, :].to(tl.float32)
    )

    q_offs = (
        r[:, None] * s_q_t
        + rk[None, :]
    )

    h_offs = (
        rv[:, None] * s_h_v
        + rk[None, :]
    )

    v_offs = (
        r[:, None] * s_v_t
        + rv[None, :]
    )

    A_offs = (
        r[:, None] * s_a_t
        + c[None, :]
    )

    for i in range(
        pid * per_core,
        min(total, (pid + 1) * per_core),
    ):
        i_t = i // (v_tiles * H)

        rem = i - i_t * (v_tiles * H)

        i_h = rem // v_tiles
        i_v = rem - i_h * v_tiles

        b_o = tl.zeros(
            [BT, BV],
            dtype=tl.float32,
        )

        b_q = tl.load(
            q
            + i_h * K
            + i_t * BT * s_q_t
            + q_offs,
        )

        b_q = (b_q * scale).to(b_q.dtype)

        b_g = tl.load(
            g
            + i_h * K
            + i_t * BT * s_q_t
            + q_offs,
        )

        b_qg = (
            b_q * tl.math.exp2(b_g)
        ).to(b_q.dtype)

        b_h = tl.load(
            h
            + (i_t * H + i_h) * V * K
            + i_v * BV * s_h_v
            + h_offs,
        )

        b_o += tl.dot(
            b_qg,
            tl.trans(b_h).to(b_qg.dtype),
        )

        b_v = tl.load(
            v
            + i_h * V
            + i_t * BT * s_v_t
            + i_v * BV
            + v_offs,
        )

        b_A = tl.load(
            A
            + i_h * BT
            + i_t * BT * s_a_t
            + A_offs,
        )

        b_A = tl.where(
            m_s,
            b_A,
            0.0,
        ).to(b_v.dtype)

        b_o += tl.dot(
            b_A,
            b_v,
        )

        tl.store(
            o
            + i_h * V
            + i_t * BT * s_v_t
            + i_v * BV
            + v_offs,
            b_o.to(o.dtype.element_ty),
        )


@triton.jit(do_not_specialize=["T"])
def chunk_gla_fwd_kernel_o_hm(
    q, v, g, h, o, A,
    scale, T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HM: tl.constexpr,
    NS: tl.constexpr,
):
    """
    Head-merged version using block pointers for all global-memory accesses.

    Grid:
        24 physical programs.

    Each CTA processes:
        (head-group, time-chunk, value-chunk)

    Memory accesses use tl.make_block_ptr + boundary_check instead of
    explicit offset vectors and masks.

    Math remains identical:
        cross-chunk:
            q_gated @ h^T

        intra-chunk:
            A_causal @ v
    """

    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)  # = 24

    NT = tl.cdiv(T, BT)
    n_v = tl.cdiv(V, BV)
    n_hg = H // HM

    total = n_v * NT * n_hg
    per_core = tl.cdiv(total, num_programs)

    # Physical strides.
    s_q_t: tl.constexpr = H * K
    s_v_t: tl.constexpr = H * V
    s_a_t: tl.constexpr = H * BT

    r = tl.arange(0, BT)
    c = tl.arange(0, BT)

    m_s = (
        r[:, None].to(tl.float32)
        >= c[None, :].to(tl.float32)
    )

    for work_id in range(
        pid * per_core,
        min(total, (pid + 1) * per_core),
    ):
        i_hg = work_id // (NT * n_v)

        rem = work_id - i_hg * NT * n_v

        i_t = rem // n_v
        i_v = rem - i_t * n_v

        # B = 1
        bos = 0

        # ------------------------------------------------------------
        # SAME HM LOOP
        # ------------------------------------------------------------
        for hh in tl.range(HM, num_stages=NS):
            i_h = i_hg * HM + hh

            b_o = tl.zeros(
                [BT, BV],
                dtype=tl.float32,
            )

            q_ptr = tl.make_block_ptr(
                base=q + (bos * H + i_h) * K,
                shape=(T, K),
                strides=(s_q_t, 1),
                offsets=(i_t * BT, 0),
                block_shape=(BT, BK),
                order=(1, 0),
            )

            b_q = tl.load(
                q_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
            )

            b_q = (b_q * scale).to(b_q.dtype)

            # g has the same physical layout as q.
            g_ptr = tl.make_block_ptr(
                base=g + (bos * H + i_h) * K,
                shape=(T, K),
                strides=(s_q_t, 1),
                offsets=(i_t * BT, 0),
                block_shape=(BT, BK),
                order=(1, 0),
            )

            b_g = tl.load(
                g_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
            )

            b_qg = (
                b_q * tl.math.exp2(b_g)
            ).to(b_q.dtype)

            h_ptr = tl.make_block_ptr(
                base=(
                    h
                    + (i_t * H + i_h) * V * K
                ),
                shape=(V, K),
                strides=(K, 1),
                offsets=(i_v * BV, 0),
                block_shape=(BV, BK),
                order=(1, 0),
            )

            b_h = tl.load(
                h_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
            )

            b_o += tl.dot(
                b_qg,
                tl.trans(b_h).to(b_qg.dtype),
            )

            v_ptr = tl.make_block_ptr(
                base=v + (bos * H + i_h) * V,
                shape=(T, V),
                strides=(s_v_t, 1),
                offsets=(i_t * BT, i_v * BV),
                block_shape=(BT, BV),
                order=(1, 0),
            )

            b_v = tl.load(
                v_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
            )

            A_ptr = tl.make_block_ptr(
                base=A + (bos * H + i_h) * BT,
                shape=(T, BT),
                strides=(s_a_t, 1),
                offsets=(i_t * BT, 0),
                block_shape=(BT, BT),
                order=(1, 0),
            )

            b_A = tl.load(
                A_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
            )

            b_A = tl.where(
                m_s,
                b_A,
                0.0,
            ).to(b_v.dtype)

            b_o += tl.dot(
                b_A,
                b_v,
            )

            o_ptr = tl.make_block_ptr(
                base=o + (bos * H + i_h) * V,
                shape=(T, V),
                strides=(s_v_t, 1),
                offsets=(i_t * BT, i_v * BV),
                block_shape=(BT, BV),
                order=(1, 0),
            )

            tl.store(
                o_ptr,
                b_o.to(o.dtype.element_ty),
                boundary_check=(0, 1),
            )


def gla_output_triton(
    q,
    v_new,
    g,
    Aqk,
    h,
    scale,
    chunk_size=_DEFAULT_BT,
    out_dtype=None,
):
    """Triton version of the KDA GLA Output operator (runs on NPU).

    Parameters are compatible with the upstream ``chunk_gla_fwd_o_gk`` (with the
    VARLEN/chunk_indices path removed):

        q:     [B, T, H, K]    query vectors (bf16/fp16/fp32)
        v_new: [B, T, H, V]    corrected values (Kernel 5 output)
        g:     [B, T, H, K]    cumulative gate (Kernel 1 output, log2 space)
        Aqk:   [B, T, H, BT]   intra-chunk causal attention weights
        h:     [B, NT, H, V, K] compressed-state snapshots (Kernel 5 output)
        scale: float           attention scale factor 1/sqrt(K)

    Returns:
        [B, T, H, V] result with the same dtype as q (bf16/fp16), on NPU.
        Falls back to the CPU reference (fp32) automatically when NPU is unavailable.
    """
    assert q.dim() == 4, f"q must be 4D [B,T,H,K], got shape {tuple(q.shape)}"
    assert h.dim() == 5, f"h must be 5D [B,NT,H,V,K], got shape {tuple(h.shape)}"

    # Pure-CPU / no-NPU environment: fall back to the reference implementation
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        return gla_output_ref(q, v_new, g, Aqk, h, scale, chunk_size=chunk_size)

    B, T, H, K = q.shape
    V = v_new.shape[-1]
    BT = int(chunk_size)
    NT = _cdiv(T, BT)

    # h's batch dim must be flattened to [B*NT, H, V, K] to match the kernel's indexing
    # (the kernel uses i_tg = i_b * NT + i_t as the first-dim index)
    h_flat = h.reshape(B * NT, H, V, K).contiguous()

    # output tensor (same dtype as q; bf16 by default)
    if out_dtype is None:
        out_dtype = q.dtype
    o = torch.empty(B, T, H, V, dtype=out_dtype, device=q.device)

    # BK=min(K,128): for the target case K=128 the 4 small dots merge into 1; K<128 uses full K.
    # num_warps: use 2 for BK=128 (best in isolated experiments), 1 for small BK.
    BK = 128 if K >= 128 else K
    BV = _DEFAULT_BV
    nw = 2 if BK >= 128 else 1

    # Scalar-addressing reduction (fifth round, 2026-08-25): the HM head-merge amortizes
    # each CTA's fixed scalar setup HM times (target case 6.94->4.75ms). Enable HM=16 when
    # H%16==0; otherwise fall back to the original kernel.
    HM = 16 if (H % 16 == 0 and BV <= V) else 1
    if HM > 1:
        # grid = (_cdiv(V, BV), NT, B * (H // HM))
        grid = (24, )
        chunk_gla_fwd_kernel_o_hm[grid](
            q=q,
            v=v_new,
            g=g,
            h=h_flat,
            o=o,
            A=Aqk,
            scale=float(scale),
            T=T,
            H=H,
            K=K,
            V=V,
            BT=BT,
            BK=BK,
            BV=BV,
            HM=HM,
            NS=_K6_NS,
            num_warps=_K6_NW,
        )
        torch.npu.synchronize()
        return o

    # grid = (_cdiv(V, BV), NT, B * H)

    grid = (24, )

    chunk_gla_fwd_kernel_o[grid](
        q=q,
        v=v_new,
        g=g,
        h=h_flat,
        o=o,
        A=Aqk,
        scale=float(scale),
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    torch.npu.synchronize()
    return o
