#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""KDA Kernel-2 (Token Parallel: diagonal Aqk/Akk) standalone implementation: pure torch + triton.

This directory does not depend on the sglang package; it only uses ``torch`` / ``triton``.
Its math is exactly identical to the upstream
``python/sglang/kernels/ops/attention/fla/chunk_intra_token_parallel.py``
(``chunk_kda_fwd_kernel_intra_token_parallel``):

    Aqk[i, j] = <q[i],  k[j] * exp2(g[i]-g[j])> * scale          (j <= i, same sub-chunk)
    Akk[i, j] = <k[i]·beta[i], k[j] * exp2(g[i]-g[j])>            (j < i,  same sub-chunk)

The module provides:
  * ``token_parallel_ref``    -- pure torch CPU reference (per-token loop, ground truth)
  * ``token_parallel_torch``  -- torch meta-operator version (**performance/precision baseline**;
                                 batched matmul per sub-chunk, avoiding a double Python loop)
  * ``token_parallel_triton`` -- triton kernel version (Route A: vectorized sub-chunk math,
                                 removing the inner Python for j loop; batched matmul via tl.dot)

Optimization notes (in this package this kernel is the final converged kernel; for the
iteration history see the companion README "Known conclusions"):
  * The original kernel was 1 CTA/token/head, grid=(B*T, H). Once flattened for a large
    case (B*T*H) it far exceeded the NPU coreDim limit of 65535, so the kernel could not launch.
  * Both Route A (this file) and Route B (token_parallel_kernel_opt_B.py) switched to a
    chunked grid = (B*cdiv(T,BT), H), eliminating the grid overflow.
  * This file takes the Route A approach: the math transform exp2(g[i]-g[j]) → exp2(g[i])*exp2(-g[j])
    batches the BC×BC gated-dot with tl.dot, removing the Python for j loop;
    aiv_scalar_ratio 0.38→0.059 (first time below the 0.10 threshold).
"""

import os

import torch
import torch_npu  # noqa: F401  (must import before creating any npu tensor)
import triton
import triton.language as tl

_BT = 64   # chunk size
_BC = 16   # sub-chunk size
# Iteration-experiment parameters (env-overridable; defaults match the converged configuration)
_K2_NW = int(os.getenv("K2_NW", "1"))
_K2_NS = int(os.getenv("K2_NS", "1"))   # number of software-pipeline stages for the head loop
_K2_HM = int(os.getenv("K2_HM", "16"))  # head-merge count (for the iteration experiments)


def _cdiv(a: int, b: int) -> int:
    return -(a // -b)


# ═══════════════════════════════════════════════════════════════════════════
# torch CPU reference (ground truth)
# ═══════════════════════════════════════════════════════════════════════════

def token_parallel_ref(q, k, gk, beta, scale, chunk_size=_BT, sub_chunk_size=_BC):
    """Per-token-loop CPU reference (verbatim math):

        Aqk[b][t][h][j % BT]  = <q[t], k[j] * exp2(g[t]-g[j])> * scale
                                 for j in [i_ts, min(t, i_ts+BC))
        Akk[b][t][h][j-i_ts]  = <k[t]·beta[t], k[j] * exp2(g[t]-g[j])>
                                 for j in [i_ts, min(t, i_ts+BC)), j < t

    Supports arbitrary B/T/H/K. When T is not a multiple of BT, trailing tokens only
    compute their own interval.
    """
    B, T, H, K = q.shape
    BT, BC = chunk_size, sub_chunk_size
    qf, kf, gf, bf = q.float(), k.float(), gk.float(), beta.float()
    Aqk = torch.zeros(B, T, H, BT, dtype=torch.float32)
    Akk = torch.zeros(B, T, H, BC, dtype=torch.float32)
    for b in range(B):
        for h in range(H):
            for t in range(T):
                i_c, i_s = t // BT, (t % BT) // BC
                i_ts = i_c * BT + i_s * BC
                qt = qf[b, t, h]
                kt = kf[b, t, h] * bf[b, t, h]
                gt = gf[b, t, h]
                for j in range(i_ts, min(t + 1, min(T, i_ts + BC))):
                    kj = kf[b, j, h]
                    gj = gf[b, j, h]
                    kgj = kj * torch.exp2(gt - gj)
                    Aqk[b, t, h, j % BT] = float((qt * kgj).sum()) * scale
                    if j < t:
                        Akk[b, t, h, j - i_ts] = float((kt * kgj).sum())
    return Aqk, Akk


# ═══════════════════════════════════════════════════════════════════════════
# torch meta-operator version (performance baseline)
# ═══════════════════════════════════════════════════════════════════════════

def token_parallel_torch(q, k, gk, beta, scale, chunk_size=_BT, sub_chunk_size=_BC):
    """Batched torch version (parallelizes over sub-chunks; one BC×BC gated-dot per sub-chunk).

    Produces the same Aqk / Akk as ``token_parallel_ref``. Uses tensor ops instead of a
    Python per-token loop and serves as the performance baseline:

        for each sub-chunk sc:
            qc, kc, gc  = [B, NT, BC, H, K] slices
            dec[i,j]    = exp2(gc[i] - gc[j])         [BC, BC]
            aqk[i,j]    = (qc[i]·(kc[j]*dec[i,j]))    [BC, BC]
            akk[i,j]    = (kbc[i]·(kc[j]*dec[i,j]))
            apply causal (j<=i) / strict (j<i) mask
            write back Aqk[.., sc*BC+j] / Akk[.., j]
    """
    B, T, H, K = q.shape
    BT, BC = chunk_size, sub_chunk_size
    NT = _cdiv(T, BT)
    NC = BT // BC
    dev = q.device
    qf, kf, gf, bf = q.float(), k.float(), gk.float(), beta.float()
    # Zero-pad the tail to NT*BT before reshape (zero-padding is consistent with the intra-chunk cumsum semantics)
    pad = NT * BT - T
    if pad:
        z = torch.zeros(B, pad, H, K, dtype=torch.float32, device=dev)
        qf = torch.cat([qf, z], dim=1)
        kf = torch.cat([kf, z], dim=1)
        gf = torch.cat([gf, z.clone()], dim=1)
        bf = torch.cat([bf, torch.zeros(B, pad, H, dtype=torch.float32, device=dev)], dim=1)
    kbc = kf * bf[..., None]
    # 5D container: [B, NT, BT, H, BT/BC]; reshape after aligning writes to sub-chunk positions
    Aqk5 = torch.zeros(B, NT, BT, H, BT, device=dev, dtype=torch.float32)
    Akk5 = torch.zeros(B, NT, BT, H, BC, device=dev, dtype=torch.float32)
    tri = torch.tril(torch.ones(BC, BC, dtype=torch.bool, device=dev))       # j<=i
    eye = torch.eye(BC, dtype=torch.bool, device=dev)                        # j==i
    for sc in range(NC):
        a = sc * BC
        qr = qf.reshape(B, NT, BT, H, K)[:, :, a : a + BC]   # [B,NT,BC,H,K]
        kr = kf.reshape(B, NT, BT, H, K)[:, :, a : a + BC]
        gr = gf.reshape(B, NT, BT, H, K)[:, :, a : a + BC]
        kbr = kbc.reshape(B, NT, BT, H, K)[:, :, a : a + BC]
        dec = torch.exp2(gr[:, :, :, None, :, :] - gr[:, :, None, :, :, :])
        kw = kr[:, :, None, :, :, :] * dec          # [B,NT,i,j,H,K]
        aqk = (qr[:, :, :, None, :, :] * kw).sum(-1)  # [B,NT,i,j,H]
        akk = (kbr[:, :, :, None, :, :] * kw).sum(-1)
        # Aqk: causal j<=i, multiply by scale;  Akk: strictly causal j<i (exclude the diagonal)
        aqk = aqk * tri[None, None, :, :, None] * scale
        akk = akk * tri[None, None, :, :, None] * (~eye)[None, None, :, :, None]
        # 5D contiguous write-back:
        #   aqk[i,j,h] -> Aqk[.., in-chunk position a+i, .., a+j]
        #   akk[i,j,h] -> Akk[.., in-chunk position a+i, .., j]   (column = offset within the sub-chunk)
        Aqk5[:, :, a : a + BC, :, a : a + BC] = aqk.permute(0, 1, 2, 4, 3)  # [B,NT,BC,H,BC]
        Akk5[:, :, a : a + BC, :, :] = akk.permute(0, 1, 2, 4, 3)
    Aqk = Aqk5.reshape(B, NT * BT, H, BT)[:, :T].contiguous()
    Akk = Akk5.reshape(B, NT * BT, H, BC)[:, :T].contiguous()
    return Aqk, Akk


# ═══════════════════════════════════════════════════════════════════════════
# triton kernel (Route C: whole-chunk big dot + contiguous write-back)
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit(do_not_specialize=["T"])
def _token_parallel_kernel(
    q, k, g, beta, Aqk, AkkScratch,
    scale,
    T, H: tl.constexpr, K: tl.constexpr,
    BT: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr,
):
    """1 CTA / (chunk, head). The whole chunk [BT,BK] in one tl.dot, diagonal-block mask.

    Route C optimization strategy (compared with Route A/B):
      * Route A did 2 small [16,128]@[128,16] dots per sub-chunk (16 tokens), 4 iterations
        in all -- the dots were too small (low cube utilization) and the per-iteration scalar
        overhead was high; measured 1243ms (6x slower than torch).
      * Route C loads the whole chunk (BT=64 rows) once and does only 2 large tl.dots
        [64,128]@[128,64] (Aqk / Akk), a single diagonal mask pass + a 2D write-back.
      * **Ascend MTE pitfall**: if the store column addresses are non-monotonic (Akk's col
        mapping c-(r//16)*16), the MTE instruction addresses go out of bounds → aicore
        exception. So Aqk is written straight to the [B,T,H,BT] output, and Akk_full is
        written to the same-layout AkkScratch [B,T,H,BT] (contiguous); the driver uses
        torch.gather to compact the diagonal 16×16 blocks into [B,T,H,16].
      * **Mask-free write-back** (msprof shows aiv_scalar 0.416 is the #1 bottleneck, coming
        from per-lane scalar addressing of the masked stores): the Aqk/Akk output buffers are
        padded to NT*BT and stores are unmasked (writing 0.0 at masked positions is enough,
        consistent with the torch ref's zero init), kernel 9.82ms vs 10.87ms.

    Math (exp2(g[i]-g[j]) = exp2(g[i])*exp2(-g[j])):
      Aqk[i,j] = (q*exp2(g)) @ (k*exp2(-g))^T * scale    (j<=i, same block)
      Akk[i,j] = (k·beta*exp2(g)) @ (k*exp2(-g))^T        (j<i,  same block)
    """
    i_cg, i_hg = tl.program_id(0), tl.program_id(1)
    NT = tl.cdiv(T, BT)
    bos = (i_cg // NT) * T
    i_c = i_cg % NT
    chunk_start = i_c * BT

    o_k = tl.arange(0, BK)
    m_k = o_k.to(tl.float32) < K
    o_r = tl.arange(0, BT)
    T_fp = T.to(tl.float32)
    chunk_start_fp = chunk_start.to(tl.float32)
    m_rows = (chunk_start_fp + o_r.to(tl.float32)) < T_fp

    base_q = q + bos * H * K + i_hg * K
    base_k = k + bos * H * K + i_hg * K
    base_g = g + bos * H * K + i_hg * K
    base_beta = beta + bos * H + i_hg
    base_aqk = Aqk + bos * H * BT + i_hg * BT
    base_akk = AkkScratch + bos * H * BT + i_hg * BT

    # ── Load the whole chunk at once ──
    row_offset = (chunk_start + o_r[:, None]) * H * K
    qc = tl.load(base_q + row_offset + o_k[None, :],
                 mask=m_rows[:, None] & m_k[None, :], other=0.0,
                 care_padding=False).to(tl.float32)
    kc = tl.load(base_k + row_offset + o_k[None, :],
                 mask=m_rows[:, None] & m_k[None, :], other=0.0,
                 care_padding=False).to(tl.float32)
    gc = tl.load(base_g + row_offset + o_k[None, :],
                 mask=m_rows[:, None] & m_k[None, :], other=0.0,
                 care_padding=False).to(tl.float32)
    betac = tl.load(base_beta + (chunk_start + o_r) * H,
                    mask=m_rows, other=0.0, care_padding=False).to(tl.float32)

    # ── Math transform + big dot ──
    qe = qc * tl.math.exp2(gc)           # [BT, BK]
    ke = kc * tl.math.exp2(-gc)          # [BT, BK]
    Aqk_full = tl.dot(qe, tl.trans(ke))  # [BT, BT]
    kbe = (kc * betac[:, None]) * tl.math.exp2(gc)
    Akk_full = tl.dot(kbe, tl.trans(ke))  # [BT, BT]

    # ── Diagonal 16×16 block mask (zero the values, not a store mask) ──
    br = o_r // BC
    bc_ = tl.arange(0, BT) // BC
    ri = o_r % BC
    ci = tl.arange(0, BT) % BC
    diag = br[:, None] == bc_[None, :]
    causal = ri[:, None] >= ci[None, :]       # j<=i
    strict = ri[:, None] > ci[None, :]        # j<i

    Aqk_full = tl.where(diag & causal, Aqk_full * scale, 0.0)
    Akk_full = tl.where(diag & strict, Akk_full, 0.0)

    # ── 2D mask-free contiguous write-back (buffers padded to NT*BT, columns monotonic 0..BT-1) ──
    r_o = chunk_start + o_r
    tl.store(base_aqk + r_o[:, None] * H * BT + tl.arange(0, BT)[None, :], Aqk_full)
    tl.store(base_akk + r_o[:, None] * H * BT + tl.arange(0, BT)[None, :], Akk_full)


@triton.jit(do_not_specialize=["T"])
def _token_parallel_kernel_hm2(
    q, k, g, beta, Aqk, AkkScratch,
    scale,
    T, H: tl.constexpr, K: tl.constexpr,
    BT: tl.constexpr, BC: tl.constexpr, HM: tl.constexpr,
):
    """head-merged version (grid=(cdiv(T,BT), B*H//HM), each CTA loops over HM heads).

    Relative to ``_token_parallel_kernel`` (1 CTA/(chunk,head), 24576 CTAs):
      * CTA count 24576→1536, amortizing the per-CTA scalar-addressing / masking cost;
      * scalar optimizations aligned with K3: drop the K-dim mask (K must be a power of 2),
        fold scale into the pre-dot q multiply, compute exp2(gc)/exp2(-gc) once each,
        and precompute the keep/strict masks outside the loop.

    Math is the same as ``_token_parallel_kernel``:
      Aqk[i,j] = <q[i], k[j]*exp2(g[i]-g[j])>*scale (j<=i, same sub-chunk)
      Akk[i,j] = <k[i]·beta[i], k[j]*exp2(g[i]-g[j])> (j<i, same sub-chunk)
    """
    i_cg, i_hg = tl.program_id(0), tl.program_id(1)
    NT = tl.cdiv(T, BT)
    n_hg = H // HM
    i_b = i_hg // n_hg
    hg0 = i_hg % n_hg
    bos = i_b * T
    i_c = i_cg % NT
    chunk_start = i_c * BT

    o_k = tl.arange(0, K)
    o_r = tl.arange(0, BT)
    T_fp = T.to(tl.float32)
    chunk_start_fp = chunk_start.to(tl.float32)
    m_rows = (chunk_start_fp + o_r.to(tl.float32)) < T_fp

    # Masks (computed once outside the loop and reused): causal / strict within the diagonal 16×16 block
    br = o_r // BC
    bc_ = tl.arange(0, BT) // BC
    ri = o_r % BC
    ci = tl.arange(0, BT) % BC
    diag = br[:, None] == bc_[None, :]
    keep = diag & (ri[:, None] >= ci[None, :])
    strict = diag & (ri[:, None] > ci[None, :])

    row_offset = (chunk_start + o_r[:, None]) * H * K
    col_bt = tl.arange(0, BT)[None, :]
    row_akk = (chunk_start + o_r[:, None]) * H * BT

    for hh in range(HM):
        i_h = hg0 * HM + hh
        base_q = q + bos * H * K + i_h * K
        base_k = k + bos * H * K + i_h * K
        base_g = g + bos * H * K + i_h * K
        base_beta = beta + bos * H + i_h
        base_aqk = Aqk + bos * H * BT + i_h * BT
        base_akk = AkkScratch + bos * H * BT + i_h * BT

        qc = tl.load(base_q + row_offset + o_k[None, :],
                     mask=m_rows[:, None], other=0.0,
                     care_padding=False).to(tl.float32)
        kc = tl.load(base_k + row_offset + o_k[None, :],
                     mask=m_rows[:, None], other=0.0,
                     care_padding=False).to(tl.float32)
        gc = tl.load(base_g + row_offset + o_k[None, :],
                     mask=m_rows[:, None], other=0.0,
                     care_padding=False).to(tl.float32)
        betac = tl.load(base_beta + (chunk_start + o_r) * H,
                        mask=m_rows, other=0.0, care_padding=False).to(tl.float32)

        eg = tl.math.exp2(gc)
        eneg = tl.math.exp2(-gc)
        qe = qc * eg * scale
        ke = kc * eneg
        Aqk_full = tl.dot(qe, tl.trans(ke))
        kbe = (kc * betac[:, None]) * eg
        Akk_full = tl.dot(kbe, tl.trans(ke))

        Aqk_full = tl.where(keep, Aqk_full, 0.0)
        Akk_full = tl.where(strict, Akk_full, 0.0)

        tl.store(base_aqk + row_akk + col_bt, Aqk_full)
        tl.store(base_akk + row_akk + col_bt, Akk_full)


@triton.jit(do_not_specialize=["T"])
def _token_parallel_kernel_hm3(
    q, k, g, beta, Aqk, AkkOut,
    scale,
    T, H: tl.constexpr, K: tl.constexpr,
    BT: tl.constexpr, BC: tl.constexpr, HM: tl.constexpr,
    NS: tl.constexpr,
):
    """head-merged v3: same as hm2, but the Akk diagonal blocks are gathered in-kernel
    via tl.gather into a compact [BT,BC] write-back to [B,T,H,BC], removing the
    full-width scratch write + driver torch.gather (msprof: gather chain ~2ms/call).
    K must be a power of 2."""
    i_cg, i_hg = tl.program_id(0), tl.program_id(1)
    NT = tl.cdiv(T, BT)
    n_hg = H // HM
    i_b = i_hg // n_hg
    hg0 = i_hg % n_hg
    bos = i_b * T
    i_c = i_cg % NT
    chunk_start = i_c * BT

    o_k = tl.arange(0, K)
    o_r = tl.arange(0, BT)
    T_fp = T.to(tl.float32)
    chunk_start_fp = chunk_start.to(tl.float32)
    m_rows = (chunk_start_fp + o_r.to(tl.float32)) < T_fp

    br = o_r // BC
    bc_ = tl.arange(0, BT) // BC
    ri = o_r % BC
    ci = tl.arange(0, BT) % BC
    diag = br[:, None] == bc_[None, :]
    keep = diag & (ri[:, None] >= ci[None, :])      # Aqk: j<=i same sub-chunk
    o_cc = tl.arange(0, BC)
    col_idx = (o_r // BC)[:, None] * BC + o_cc[None, :]   # [BT,BC] row-dependent column offsets
    strict = (o_r % BC)[:, None] > o_cc[None, :]          # Akk: j<i within the block

    row_offset = (chunk_start + o_r[:, None]) * H * K
    col_bt = tl.arange(0, BT)[None, :]
    row_aqk = (chunk_start + o_r[:, None]) * H * BT
    row_akk = (chunk_start + o_r[:, None]) * H * BC

    for hh in tl.range(HM, num_stages=NS):
        i_h = hg0 * HM + hh
        base_q = q + bos * H * K + i_h * K
        base_k = k + bos * H * K + i_h * K
        base_g = g + bos * H * K + i_h * K
        base_beta = beta + bos * H + i_h
        base_aqk = Aqk + bos * H * BT + i_h * BT
        base_akk = AkkOut + bos * H * BC + i_h * BC

        qc = tl.load(base_q + row_offset + o_k[None, :],
                     mask=m_rows[:, None], other=0.0,
                     care_padding=False).to(tl.float32)
        kc = tl.load(base_k + row_offset + o_k[None, :],
                     mask=m_rows[:, None], other=0.0,
                     care_padding=False).to(tl.float32)
        gc = tl.load(base_g + row_offset + o_k[None, :],
                     mask=m_rows[:, None], other=0.0,
                     care_padding=False).to(tl.float32)
        betac = tl.load(base_beta + (chunk_start + o_r) * H,
                        mask=m_rows, other=0.0, care_padding=False).to(tl.float32)

        eg = tl.math.exp2(gc)
        eneg = tl.math.exp2(-gc)
        qe = qc * eg * scale
        ke = kc * eneg
        Aqk_full = tl.dot(qe, tl.trans(ke))
        kbe = (kc * betac[:, None]) * eg
        Akk_full = tl.dot(kbe, tl.trans(ke))

        Aqk_full = tl.where(keep, Aqk_full, 0.0)
        tl.store(base_aqk + row_aqk + col_bt, Aqk_full)

        Akk_diag = tl.gather(Akk_full, col_idx, axis=1)   # [BT,BC]
        Akk_diag = tl.where(strict, Akk_diag, 0.0)
        tl.store(base_akk + row_akk + o_cc[None, :], Akk_diag)


def _gather_akk_diag(scratch, BC, T=None):
    """Collect the diagonal 16×16 blocks from the [B,TP,H,BT] scratch into [B,T,H,BC].

    The scratch may be padded to NT*BT (required for the unmasked stores); trim back to the
    real T before returning.
    """
    B, TP, H, BT = scratch.shape
    NT = TP // BT
    if T is None:
        T = TP
    dev = scratch.device
    r = torch.arange(BT, device=dev)
    idx = (r.view(1, 1, BT, 1, 1) // BC * BC +
           torch.arange(BC, device=dev).view(1, 1, 1, 1, BC))
    idx = idx.expand(B, NT, BT, H, BC)          # [B,NT,BT,H,BC]
    scr = scratch.reshape(B, NT, BT, H, BT)     # view
    Akk = torch.gather(scr, 4, idx)             # [B,NT,BT,H,BC]
    return Akk.reshape(B, NT * BT, H, BC)[:, :T].contiguous()


def token_parallel_triton(
    q, k, gk, beta, scale,
    Aqk=None, Akk=None,
    chunk_size=_BT, sub_chunk_size=_BC,
):
    """triton kernel version (Route C: whole-chunk big dot + mask-free write-back). Returns (Aqk, Akk).

    Unlike the old implementation that delegated to torch_npu, this is a **real triton kernel**:
      * grid = (B*cdiv(T,BT), H), one (chunk, head) per CTA; 2 large tl.dots over the whole
        chunk, diagonal-block mask, mask-free contiguous write-back (buffer padded to NT*BT).
      * The Akk diagonal blocks are collected by the driver with torch.gather (the MTE
        column mapping would go out of bounds).
    """
    B, T, H, K = q.shape
    BT, BC = chunk_size, sub_chunk_size
    BK = triton.next_power_of_2(K)
    NT = _cdiv(T, BT)
    TP = NT * BT
    dev = q.device
    # The kernel does a mask-free full-width write-back (buffer padded to NT*BT, writing 0.0 at masked positions), so no pre-zeroing is needed.
    # empty avoids an extra 800MB memset kernel per call (inside the timed region).
    if Aqk is None:
        Aqk = torch.empty(B, TP, H, BT, device=dev, dtype=torch.float32)
    # head-merge fast path (hm3): when K is a power of 2, tl.arange(0,K) needs no K mask, and
    # the Akk diagonal blocks are gathered in-kernel via tl.gather into a compact [B,T,H,BC]
    # output, eliminating the full-width scratch write + driver torch.gather (~2ms/call).
    # HM=16 shrinks grid dim 2 from B*H to B*H//16 (24576→1536 CTAs). If H%16!=0, HM=1
    # degrades to the same 1 CTA/(chunk,head) structure as the original kernel.
    if K == BK:
        HM = _K2_HM if H % _K2_HM == 0 else 1
        grid = (NT, B * (H // HM))
        if Akk is None:
            Akk = torch.empty(B, TP, H, BC, device=dev, dtype=torch.float32)
        _token_parallel_kernel_hm3[grid](
            q, k, gk, beta, Aqk, Akk, float(scale),
            T, H=H, K=K, BT=BT, BC=BC, HM=HM, NS=_K2_NS,
            num_warps=_K2_NW,
        )
        torch.npu.synchronize()
        return Aqk[:, :T], Akk[:, :T]
    # Fallback path: K not a power of 2 → original _token_parallel_kernel + torch.gather collection
    scratch = torch.empty(B, TP, H, BT, device=dev, dtype=torch.float32)
    grid = (B * NT, H)
    _token_parallel_kernel[grid](
        q, k, gk, beta, Aqk, scratch, float(scale),
        T, H=H, K=K, BT=BT, BC=BC, BK=BK, num_warps=1,
    )
    torch.npu.synchronize()
    Aqk = Aqk[:, :T]
    if Akk is None:
        Akk = _gather_akk_diag(scratch, BC, T=T)
    else:
        Akk.copy_(_gather_akk_diag(scratch, BC, T=T))
    return Aqk, Akk
