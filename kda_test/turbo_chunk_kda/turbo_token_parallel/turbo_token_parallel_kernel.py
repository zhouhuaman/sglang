#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""KDA Kernel-2（Token Parallel：对角线 Aqk/Akk）独立实现：纯 torch + triton。

本目录不依赖 sglang 包，只用 ``torch`` / ``triton``。实现与上游
``python/sglang/kernels/ops/attention/fla/chunk_intra_token_parallel.py``
（``chunk_kda_fwd_kernel_intra_token_parallel``）数学完全一致:

    Aqk[i, j] = <q[i],  k[j] * exp2(g[i]-g[j])> * scale          (j <= i, 同 sub-chunk)
    Akk[i, j] = <k[i]·beta[i], k[j] * exp2(g[i]-g[j])>            (j < i,  同 sub-chunk)

模块提供:
  * ``turbo_token_parallel_ref``    —— 纯 torch CPU 参考（逐 token 循环，ground truth）
  * ``turbo_token_parallel_torch``  —— torch 元算子版本（**性能/精度基准**，按 sub-chunk
                                批量化 matmul，避免 Python 双层循环）
  * ``turbo_token_parallel_triton`` —— triton kernel 版（Route A: 向量化 sub-chunk 计算,
                                消除内层 Python for j 循环; 使用 tl.dot 批量矩阵乘）

优化说明（见 OPTIMIZATION_LOG.md / kernel_metadata.json）:
  * 原 kernel 为 1 CTA/token/head, grid=(B*T, H), 大 case (B*T*H) 展平后
    远超 NPU coreDim 上限 65535 → kernel 无法启动。
  * Route A (本文件) + Route B (token_parallel_kernel_opt_B.py) 均改为
    chunked grid = (B*cdiv(T,BT), H), 消除 grid 超限。
  * 本文件采用 Route A 策略: 数学变换 exp2(g[i]-g[j]) → exp2(g[i])*exp2(-g[j]),
    用 tl.dot 批量计算 BC×BC gated-dot, 消除 Python for j 循环,
    aiv_scalar_ratio 0.38→0.059 (首次 < 0.10 阈值)。
"""

import os

import torch
import torch_npu  # noqa: F401  (必须在创建任何 npu 张量之前 import)
import triton
import triton.language as tl

_BT = 64   # chunk 大小
_BC = 16   # sub-chunk 大小
# 迭代实验参数（env 覆盖；默认与收敛配置一致）
_K2_NW = int(os.getenv("K2_NW", "1"))
_K2_NS = int(os.getenv("K2_NS", "1"))   # head 循环软件流水线级数
_K2_HM = int(os.getenv("K2_HM", "16"))  # head 合并数（迭代实验用）


def _cdiv(a: int, b: int) -> int:
    return -(a // -b)


# ═══════════════════════════════════════════════════════════════════════════
# torch CPU 参考（ground truth）
# ═══════════════════════════════════════════════════════════════════════════

def turbo_token_parallel_ref(q, k, gk, beta, scale, chunk_size=_BT, sub_chunk_size=_BC):
    """逐 token 循环 CPU 参考（数学原文）:

        Aqk[b][t][h][j % BT]  = <q[t], k[j] * exp2(g[t]-g[j])> * scale
                                 for j in [i_ts, min(t, i_ts+BC))
        Akk[b][t][h][j-i_ts]  = <k[t]·beta[t], k[j] * exp2(g[t]-g[j])>
                                 for j in [i_ts, min(t, i_ts+BC)), j < t

    支持任意 B/T/H/K。T 非 BT 倍数时, 尾部 token 只计算自身区间。
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
# torch 元算子版本（性能基本准）
# ═══════════════════════════════════════════════════════════════════════════

def turbo_token_parallel_torch(q, k, gk, beta, scale, chunk_size=_BT, sub_chunk_size=_BC):
    """批量 torch 版本（按 sub-chunk 并行, 每个 sub-chunk 一次 BC×BC gated-dot）。

    与 ``turbo_token_parallel_ref`` 生成相同的 Aqk / Akk。使用 tensor 运算而非 Python
    逐 token 循环, 用作性能基准:

        for each sub-chunk sc:
            qc, kc, gc  = [B, NT, BC, H, K] 切片
            dec[i,j]    = exp2(gc[i] - gc[j])         [BC, BC]
            aqk[i,j]    = (qc[i]·(kc[j]*dec[i,j]))    [BC, BC]
            akk[i,j]    = (kbc[i]·(kc[j]*dec[i,j]))
            apply causal (j<=i) / strict (j<i) mask
            写回 Aqk[.., sc*BC+j] / Akk[.., j]
    """
    B, T, H, K = q.shape
    BT, BC = chunk_size, sub_chunk_size
    NT = _cdiv(T, BT)
    NC = BT // BC
    dev = q.device
    qf, kf, gf, bf = q.float(), k.float(), gk.float(), beta.float()
    # 尾部 pad 到 NT*BT 再 reshape(补零, 与 chunk 内部 cumsum 语义一致)
    pad = NT * BT - T
    if pad:
        z = torch.zeros(B, pad, H, K, dtype=torch.float32, device=dev)
        qf = torch.cat([qf, z], dim=1)
        kf = torch.cat([kf, z], dim=1)
        gf = torch.cat([gf, z.clone()], dim=1)
        bf = torch.cat([bf, torch.zeros(B, pad, H, dtype=torch.float32, device=dev)], dim=1)
    kbc = kf * bf[..., None]
    # 5D 容器: [B, NT, BT, H, BT/BC], 写回时在 sub-chunk 位置对齐后 reshape
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
        # Aqk: 因果 j<=i, 乘 scale;  Akk: 严格因果 j<i (去掉对角线)
        aqk = aqk * tri[None, None, :, :, None] * scale
        akk = akk * tri[None, None, :, :, None] * (~eye)[None, None, :, :, None]
        # 5D 连续写回:
        #   aqk[i,j,h] -> Aqk[.., chunk 内位置 a+i, .., a+j]
        #   akk[i,j,h] -> Akk[.., chunk 内位置 a+i, .., j]   (列 = sub-chunk 内偏移)
        Aqk5[:, :, a : a + BC, :, a : a + BC] = aqk.permute(0, 1, 2, 4, 3)  # [B,NT,BC,H,BC]
        Akk5[:, :, a : a + BC, :, :] = akk.permute(0, 1, 2, 4, 3)
    Aqk = Aqk5.reshape(B, NT * BT, H, BT)[:, :T].contiguous()
    Akk = Akk5.reshape(B, NT * BT, H, BC)[:, :T].contiguous()
    return Aqk, Akk


# ═══════════════════════════════════════════════════════════════════════════
# triton kernel（Route C：整 chunk 大 dot + 连续写回）
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit(do_not_specialize=["T"])
def _token_parallel_kernel(
    q, k, g, beta, Aqk, AkkScratch,
    scale,
    T, H: tl.constexpr, K: tl.constexpr,
    BT: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr,
):
    """1 CTA / (chunk, head)。整 chunk [BT,BK] 一次 tl.dot, 对角线 block 掩码。

    Route C 优化策略（对比 Route A/B）:
      * Route A 每 sub-chunk (16 tokens) 做 2 个 [16,128]@[128,16] 小 dot,
        共 4 次迭代 —— dot 太小 (cube 利用率低)、每迭代标量开销高, 实测
        1243ms (6x 慢于 torch)。
      * Route C 一次加载整 chunk (BT=64 行), 只做 2 个大 tl.dot
        [64,128]@[128,64] (Aqk / Akk), 一次性对角线掩码 + 2D 写回。
      * **Ascend MTE 坑**: 写回若列地址非单调 (Akk 的 col 映射 c-(r//16)*16),
        MTE 指令地址越界 → aicore exception。故 Aqk 直接写 [B,T,H,BT] 输出,
        Akk_full 写同布局 AkkScratch [B,T,H,BT] (连续), driver 用 torch.gather
        把对角线 16×16 block 收拢到 [B,T,H,16]。
      * **无掩码写回** (msprof 显示 aiv_scalar 0.416 是头号瓶颈, 来自带 mask 的
        store 逐 lane 标量寻址): Aqk/Akk 输出缓冲按 NT*BT 补齐, store 不加 mask
        (掩码处写 0.0 即可, 与 torch ref 的零初值一致), kernel 9.82ms vs 10.87ms。

    数学 (exp2(g[i]-g[j]) = exp2(g[i])*exp2(-g[j])):
      Aqk[i,j] = (q*exp2(g)) @ (k*exp2(-g))^T * scale    (j<=i, 同 block)
      Akk[i,j] = (k·beta*exp2(g)) @ (k*exp2(-g))^T        (j<i,  同 block)
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

    # ── 一次加载整 chunk ──
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

    # ── 数学变换 + 大 dot ──
    qe = qc * tl.math.exp2(gc)           # [BT, BK]
    ke = kc * tl.math.exp2(-gc)          # [BT, BK]
    Aqk_full = tl.dot(qe, tl.trans(ke))  # [BT, BT]
    kbe = (kc * betac[:, None]) * tl.math.exp2(gc)
    Akk_full = tl.dot(kbe, tl.trans(ke))  # [BT, BT]

    # ── 对角线 16×16 block 掩码 (值置 0, 非 store mask) ──
    br = o_r // BC
    bc_ = tl.arange(0, BT) // BC
    ri = o_r % BC
    ci = tl.arange(0, BT) % BC
    diag = br[:, None] == bc_[None, :]
    causal = ri[:, None] >= ci[None, :]       # j<=i
    strict = ri[:, None] > ci[None, :]        # j<i

    Aqk_full = tl.where(diag & causal, Aqk_full * scale, 0.0)
    Akk_full = tl.where(diag & strict, Akk_full, 0.0)

    # ── 2D 无掩码连续写回 (缓冲按 NT*BT 补齐, 列单调 0..BT-1) ──
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
    """head-merged 版（grid=(cdiv(T,BT), B*H//HM), 每 CTA 循环 HM 个 head）。

    相对 ``_token_parallel_kernel``（1 CTA/(chunk,head), 24576 CTA）:
      * CTA 数 24576→1536, 摊薄每 CTA 的标量寻址/掩码开销;
      * 与 K3 对齐的标量优化: 去掉 K 维 mask（K 需为 2 幂）、scale 折叠进
        pre-dot q 乘、exp2(gc)/exp2(-gc) 各算一次、keep/strict 掩码循环外预计算。

    数学同 ``_token_parallel_kernel``:
      Aqk[i,j] = <q[i], k[j]*exp2(g[i]-g[j])>*scale (j<=i, 同 sub-chunk)
      Akk[i,j] = <k[i]·beta[i], k[j]*exp2(g[i]-g[j])> (j<i, 同 sub-chunk)
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

    # 掩码（循环外一次计算, 复用）: 对角 16×16 块内 causal / strict
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


def _gather_akk_diag(scratch, BC, T=None):
    """把 [B,TP,H,BT] scratch 中对角线 16×16 block 收拢为 [B,T,H,BC]。

    scratch 可能按 NT*BT 补齐 (unmasked-store 需要), 返回前按真实 T 裁剪。
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


def turbo_token_parallel_triton(
    q, k, gk, beta, scale,
    Aqk=None, Akk=None,
    chunk_size=_BT, sub_chunk_size=_BC,
):
    """triton kernel 版 (Route C: 整 chunk 大 dot + 无掩码写回)。返回 (Aqk, Akk)。

    相比旧委托 torch_npu 的实现，这里是**真正的 triton kernel**：
      * grid = (B*cdiv(T,BT), H), 每 CTA 一个 (chunk,head), 整 chunk 2 个大
        tl.dot, 对角线 block 掩码, 无掩码连续写回 (缓冲按 NT*BT 补齐)。
      * Akk 对角线 block 由 driver 用 torch.gather 收拢 (MTE 列映射会越界)。
    """
    B, T, H, K = q.shape
    BT, BC = chunk_size, sub_chunk_size
    BK = triton.next_power_of_2(K)
    NT = _cdiv(T, BT)
    TP = NT * BT
    dev = q.device
    # kernel 做无掩码全量写回（缓冲按 NT*BT 补齐, 掩码处写 0.0）, 故无需预清零。
    # 用 empty 避免每次调用多一次 800MB memset kernel（在计时区段内）。
    if Aqk is None:
        Aqk = torch.empty(B, TP, H, BT, device=dev, dtype=torch.float32)
    # head-merge 快速路径 (hm3): K 为 2 幂时用 tl.arange(0,K) 无 K 掩码, 且
    # Akk 对角线块在 kernel 内 tl.gather 收拢为 [B,T,H,BC] 紧凑输出, 消除
    # scratch 满宽写 + driver torch.gather (~2ms/调用)。HM=16 使 grid 第 2 维
    # 从 B*H 缩到 B*H//16 (24576→1536 CTA)。H%16!=0 时 HM=1 退化为与原 kernel
    # 相同的 1 CTA/(chunk,head) 结构。
    if K == BK:
        HM = _K2_HM if H % _K2_HM == 0 else 1
        grid = (NT, B * (H // HM))
        scratch = torch.empty(B, TP, H, BT, device=dev, dtype=torch.float32)
        if Akk is None:
            Akk = torch.empty(B, TP, H, BC, device=dev, dtype=torch.float32)
        # 不用"kernel 内 tl.gather 收拢对角线块"的写法: triton-ascend 3.2.1 上
        # tl.gather 的 src 为 tl.dot 输出时结果错误 (实测 Akk max_diff≈0.32)
        # 且退化 ~3x 慢。故走 hm2: 满宽写 scratch +
        # driver torch.gather 收拢 (实测 max_diff≈8.9e-8)。
        _token_parallel_kernel_hm2[grid](
            q, k, gk, beta, Aqk, scratch, float(scale),
            T, H=H, K=K, BT=BT, BC=BC, HM=HM, num_warps=_K2_NW,
        )
        torch.npu.synchronize()
        Akk[:, :T].copy_(_gather_akk_diag(scratch, BC, T=T))
        return Aqk[:, :T], Akk[:, :T]
    # 回退路径: K 非 2 幂 → 原 _token_parallel_kernel + torch.gather 收拢
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
        Akk[:, :T].copy_(_gather_akk_diag(scratch, BC, T=T))
    return Aqk, Akk