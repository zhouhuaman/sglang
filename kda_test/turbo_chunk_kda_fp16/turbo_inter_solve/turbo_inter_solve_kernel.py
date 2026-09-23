#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""KDA Kernel-3（Inter-Solve Fused：非对角线 Aqk/Akk + 对角块前向替换 + 链式求逆）独立实现。

本目录不依赖 sglang 包，只用 ``torch`` / ``torch_npu`` / ``triton``。数学与上游
``python/sglang/kernels/ops/attention/fla/chunk_intra.py``
（``chunk_kda_fwd_kernel_inter_solve_fused``）一致，但做如下**独立子集化**：

  * 只做固定长度（B,T,H,K），不做 VARLEN / safe-gate / FUSE_RECOMPUTE / FUSE_DIAGONAL；
  * 输入的对角线 Akk 块已由 Kernel-2（token_parallel）写好，本 kernel 读 ``Akkd``
    直接做前向替换 + 链式求逆；
  * 输出 Akk_inv = [B, T, H, BT]（10 个 16×16 子块的合并下三角逆）。

计算内容:

    Phase 1（非对角线块, 每 CTA 一个 (chunk, head)）:
        Akk_ij = (K_i * exp2(G_i - G_i[last])) @ (K_j * exp2(G_i[last] - G_j))^T * beta_j
        Aqk_ij = (Q_i * exp2(G_i - G_i[last])) @ (K_j * exp2(G_i[last] - G_j))^T * scale
          (i, j = 0..3, i > j;  G_i[last] 是子块 i 的**末尾** token 的 g 参考点)

    Phase 2（对 4 个对角 16×16 下三角子块做逐行前向替换求逆）:
        D_inv = (I - tril(D))^{-1}   (逐行累加, fp32)

    Phase 3（链式矩阵乘合并逆）:
        Ai_10 = -Ai_11 @ Akk_10 @ Ai_00
        Ai_21 = -Ai_22 @ Akk_21 @ Ai_11
        Ai_32 = -Ai_33 @ Akk_32 @ Ai_22
        Ai_20 = -Ai_22 @ (Akk_20 @ Ai_00 + Akk_21 @ Ai_10)
        Ai_31 = -Ai_33 @ (Akk_31 @ Ai_11 + Akk_32 @ Ai_21)
        Ai_30 = -Ai_33 @ (Akk_30 @ Ai_00 + Akk_31 @ Ai_10 + Akk_32 @ Ai_20)

模块提供:
  * ``turbo_inter_solve_ref``     —— 纯 torch CPU 参考（逐 chunk 循环, ground truth）,
                                返回 (Aqk, Akk_inv)
  * ``turbo_inter_solve_torch``   —— torch 元算子版本（**精度/性能基准**）,数学与 ref 一致
  * ``turbo_inter_solve_triton``  —— triton kernel 版（1 CTA / chunk / head）
"""

import os

import torch
import torch_npu  # noqa: F401  (必须在创建任何 npu 张量之前 import)
import triton
import triton.language as tl

_BT = 64   # chunk 大小
_BC = 16   # sub-chunk 大小
# 迭代实验参数（env 覆盖；默认与收敛配置一致）
_NUM_STAGES = int(os.getenv("K3_NS", "1"))   # head 循环软件流水线级数
_NUM_WARPS = int(os.getenv("K3_NW", "4"))
# NP=2（4 dot）比 NP=3（6 dot）快 18.5%（8.71 vs 10.69 ms，目标 case，
# triton-ascend 3.2.1 口径）。精度：fp16/bf16 树 NP=2 与 NP=3 的 max_diff
# 完全相同（4.977e-04 / 3.609e-03，误差由输入量化主导，截断级数不贡献）；
# 仅 fp32 树由 4.768e-07 变为 3.311e-04，距 1e-2 门槛仍有 30x 余量。
# NP=1 精度不达标（4.463e-02），不可用。
_NP = int(os.getenv("K3_NP", "2"))           # 逆截断级数
# HM 采用交付默认 16（H % 16 == 0 时按 head 合并；H 不整除时 kernel 内回退 HM=1）。
# 注意：HM=1 可编译但更慢（NP=2 下 11.21 vs 8.71 ms，目标 case，triton-ascend 3.2.1），
# 不要改回 1。（早些时候"HM>1 编译失败"的结论来自 triton-ascend 3.2.2+dev，非交付口径。）
_HM = int(os.getenv("K3_HM", "16"))           # head 合并数（迭代实验用）


def _cdiv(a: int, b: int) -> int:
    return -(a // -b)


# ═══════════════════════════════════════════════════════════════════════════
# torch CPU 参考（ground truth）
# ═══════════════════════════════════════════════════════════════════════════

def diag_solve_forward_ref(D):
    """对一个 [n, n] 下三角矩阵做前向替换求逆（含对角线）。

    采用与上游一致的行累积算法:
        A = -strict_tril(D)
        for i in 2..n-1:
            A[i] = -D[i] + sum_k A[i,k] * A[k]     (仅 i 行, 逐行)
        D_inv = A + I

    返回 [n, n] fp32。n 任意（用于大块验算）。
    """
    n = D.shape[-1]
    A = -torch.tril(D, diagonal=-1).float()  # [n,n]
    for i in range(2, n):
        row = A[i]  # [n]
        row = row + (row[:, None] * A).sum(dim=0)  # 行乘整行累加
        A[i] = row
    return A + torch.eye(n, dtype=torch.float32, device=D.device)


def turbo_inter_solve_ref(
    q, k, g, beta, Akkd, scale,
    chunk_size=_BT, sub_chunk_size=_BC,
):
    """逐 chunk 循环的 CPU 参考。

    输入:
      q/k/g:  [B, T, H, K] fp32
      beta:   [B, T, H] fp32
      Akkd:   [B, T, H, BC] fp32  （Kernel-2 输出的对角线 Akk 块；j<i 非零，j>=i 为 0）

    输出:
      Aqk:     [B, T, H, BT]  非对角线 Aqk 子块（行=token, 列=chunk 内 j 位置）
      Akk_inv: [B, T, H, BT]  10 个 16×16 子块合并的下三角逆
    """
    B, T, H, K = q.shape
    BT, BC = chunk_size, sub_chunk_size
    NC = BT // BC
    NT = _cdiv(T, BT)
    qf, kf, gf = q.float(), k.float(), g.float()
    beta_f = beta.float()
    Akkd_f = Akkd.float()
    # 尾 chunk 不满 BT 时, 补 0 到完整 chunk 边界, 避免 sub-chunk 切片不等长
    pad = NT * BT - T
    if pad:
        zp = torch.zeros(B, pad, H, K, dtype=torch.float32)
        qf = torch.cat([qf, zp], dim=1)
        kf = torch.cat([kf, zp.clone()], dim=1)
        gf = torch.cat([gf, zp.clone()], dim=1)
        zA = torch.zeros(B, pad, H, BC, dtype=torch.float32)
        Akkd_f = torch.cat([Akkd_f, zA], dim=1)
        zb = torch.zeros(B, pad, H, dtype=torch.float32)
        beta_f = torch.cat([beta_f, zb], dim=1)
    Aqk = torch.zeros(B, NT * BT, H, BT, dtype=torch.float32)
    Akk_inv = torch.zeros(B, NT * BT, H, BT, dtype=torch.float32)
    for b in range(B):
        for h in range(H):
            for i_c in range(NT):
                base = i_c * BT
                # —— Phase 1: 非对角线块 ——
                off = {}
                for i in range(NC):
                    r = base + i * BC
                    q_i = qf[b, r:r + BC, h]          # [BC, K]
                    k_i = kf[b, r:r + BC, h]
                    g_i = gf[b, r:r + BC, h]
                    # 参考点 = 子块末尾 token 的 gate
                    gni = gf[b, r + BC - 1, h]  # [K]
                    gqn = torch.exp2(g_i - gni.unsqueeze(0))  # [BC,K]
                    for j in range(i):
                        c = base + j * BC
                        k_j = kf[b, c:c + BC, h]
                        g_j = gf[b, c:c + BC, h]
                        # 行乘 beta_i (子块 i 的 beta, 与上游 / triton kernel 一致)
                        beta_i = beta_f[b, r:r + BC, h]        # [BC]
                        # [K, BC] = (k_j * exp2(gni - g_j))^T
                        kgt = (k_j * torch.exp2(gni.unsqueeze(0) - g_j)).t()
                        bj = (q_i * gqn) @ kgt
                        Aqk[b, r:r + BC, h, c - base:c - base + BC] = bj * scale
                        bk = (k_i * gqn) @ kgt
                        off[(i, j)] = bk * beta_i[:, None]
                # —— Phase 2: 对角块前向替换 ——
                di = {}
                for i in range(NC):
                    r = base + i * BC
                    D = Akkd_f[b, r:r + BC, h].clone()  # [BC,BC]
                    di[i] = diag_solve_forward_ref(D)
                # —— Phase 3: 链式合并 ——
                Ai = {}
                for i in range(NC):
                    Ai[(i, i)] = di[i]
                for (i, j) in [(1, 0), (2, 1), (3, 2)]:
                    Ai[(i, j)] = -di[i] @ off[(i, j)] @ di[j]
                Ai[(2, 0)] = -di[2] @ (off[(2, 0)] @ di[0] + off[(2, 1)] @ Ai[(1, 0)])
                Ai[(3, 1)] = -di[3] @ (off[(3, 1)] @ di[1] + off[(3, 2)] @ Ai[(2, 1)])
                Ai[(3, 0)] = -di[3] @ (off[(3, 0)] @ di[0]
                                       + off[(3, 1)] @ Ai[(1, 0)]
                                       + off[(3, 2)] @ Ai[(2, 0)])
                # 写回 Akk_inv
                for (i, j) in [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2),
                               (3, 0), (3, 1), (3, 2), (3, 3)]:
                    Akk_inv[b, base + i * BC:base + i * BC + BC, h,
                            j * BC:j * BC + BC] = Ai[(i, j)]
    Aqk = Aqk[:, :T].contiguous()
    Akk_inv = Akk_inv[:, :T].contiguous()
    return Aqk, Akk_inv

# ═══════════════════════════════════════════════════════════════════════════
# torch 元算子版本（性能/精度基准）——按 chunk 批量化, 避免 Python 逐 chunk 循环
# ═══════════════════════════════════════════════════════════════════════════

def _batch_forward_solve(Dsub):
    """对一批 [P, BC, BC] 下三角矩阵做前向替换求逆（逐行向量化）。

    A = -strict_tril(D);  for i=2..BC-1: A[:,i] += A[:,i][:,None]*A 的总和行
    """
    P = Dsub.shape[0]
    A = -torch.tril(Dsub, diagonal=-1)          # [P,BC,BC]
    for i in range(2, Dsub.shape[1]):
        # A[:, i] = -D[:, i] + sum_k A[:, i, k] * A[:, k, :]
        row = A[:, i]                            # [P,BC]
        contrib = (row[:, :, None] * A).sum(1)   # [P,BC]
        A[:, i] = row + contrib
    I = torch.eye(Dsub.shape[1], dtype=Dsub.dtype, device=Dsub.device)
    return A + I


def turbo_inter_solve_torch(
    q, k, g, beta, Akkd, scale,
    chunk_size=_BT, sub_chunk_size=_BC,
):
    """torch 元算子版本: 与 ``turbo_inter_solve_ref`` 数学一致, 但按 chunk 批量化。

    返回值与 ref 相同: (Aqk[B,T,H,BT], Akk_inv[B,T,H,BT])。

    说明: 前向替换阶段与 Kernel-3 一样是串行的 (BC 行), 批量化收益主要来自
    Phase 1 的 [BC,BC] 块聚合与 Phase 3 的链式 matmul。
    """
    B, T, H, K = q.shape
    BT, BC = chunk_size, sub_chunk_size
    NC = BT // BC
    NT = _cdiv(T, BT)
    dev = q.device
    pad = NT * BT - T
    qf, kf, gf = q.float(), k.float(), g.float()
    beta_f = beta.float()
    Akkd_ = Akkd.float()
    if pad:
        z = torch.zeros(B, pad, H, K, dtype=torch.float32, device=dev)
        qf = torch.cat([qf, z], dim=1); kf = torch.cat([kf, z], dim=1)
        gf = torch.cat([gf, z.clone()], dim=1)
        zk = torch.zeros(B, pad, H, BC, dtype=torch.float32, device=dev)
        Akkd_ = torch.cat([Akkd_, zk], dim=1)
        zb = torch.zeros(B, pad, H, dtype=torch.float32, device=dev)
        beta_f = torch.cat([beta_f, zb], dim=1)
    # reshape to [B, NT, BT, H, K] 按 chunk 排列, 再取子块
    Q = qf.reshape(B, NT, BT, H, K)
    Ks = kf.reshape(B, NT, BT, H, K)
    G = gf.reshape(B, NT, BT, H, K)
    Bt = beta_f.reshape(B, NT, BT, H)
    D = Akkd_.reshape(B, NT, BT, H, BC)
    qi = [Q[:, :, i * BC:(i + 1) * BC, :, :] for i in range(NC)]
    ki = [Ks[:, :, i * BC:(i + 1) * BC, :, :] for i in range(NC)]
    gi = [G[:, :, i * BC:(i + 1) * BC, :, :] for i in range(NC)]
    bi = [Bt[:, :, i * BC:(i + 1) * BC, :] for i in range(NC)]
    # 参考点: 子块 i 的末尾 token 的 gate（在 chunk 内的位置 i*BC+BC-1）
    gni = [G[:, :, i * BC + BC - 1, :, :].unsqueeze(2) for i in range(NC)]  # [B,NT,1,H,K]
    # 非对角线块
    Aqk = torch.zeros(B, NT * BT, H, BT, dtype=torch.float32, device=dev)
    Akk_inv = torch.zeros(B, NT * BT, H, BT, dtype=torch.float32, device=dev)
    off = {}
    for i in range(1, NC):
        gq = torch.exp2(gi[i] - gni[i])                     # [B,NT,BC,H,K]
        for j in range(i):
            # bk_t = k_j * exp2(gni[i] - g_j)  [B,NT,BC,H,K]
            bk_t = (ki[j] * torch.exp2(gni[i] - gi[j]))
            # Aqk_ij = (q_i * gq) · bk_t^T  消 K 维, 保留两个不同的 BC 轴
            #   qi[i]*gq: [B,NT,BC_i,H,K];  bk_t: [B,NT,BC_j,H,K]
            #   einsum: 'bnihk,bnjhk->bnihj'  (i=rows, j=cols, 不可合并 d)
            aqk_m = torch.einsum('bnihk,bnjhk->bnihj', qi[i] * gq, bk_t) * scale
            akk_m = torch.einsum('bnihk,bnjhk->bnihj', ki[i] * gq, bk_t)
            # 乘 beta_i (行广播, 子块 i 的 beta, 与上游 triton kernel 一致)
            beta_i = bi[i]                          # [B,NT,BC,H]
            akk_m = akk_m * beta_i.unsqueeze(-1)
            # 写回 Aqk[b, nt*BT + i*BC + p, h, j*BC + q]
            a_i, a_j = i * BC, j * BC
            # aqk_m / akk_m: [B,NT,BC_i,H,BC_j] -> [B,NT,BC_i,H,BC_j]
            Aqk.view(B, NT, BT, H, BT)[:, :, a_i:a_i + BC, :, a_j:a_j + BC] = aqk_m
            # off[(i,j)] 统一存成 [B,NT,H,BC_i,BC_j] 以便后续链式 matmul
            off[(i, j)] = akk_m.permute(0, 1, 3, 2, 4).contiguous()  # [B,NT,H,BC,BC]
    # 对角线前向替换
    di = {}
    for i in range(NC):
        Di = D[:, :, i * BC:(i + 1) * BC, :, :]  # [B,NT,BC,H,BC]
        Di = Di.permute(0, 1, 3, 2, 4).reshape(B * NT * H, BC, BC)  # [P,BC,BC]
        di[i] = _batch_forward_solve(Di).reshape(B, NT, H, BC, BC)
    # 链式合并
    Ai = {}
    Ai[(0, 0)] = di[0]
    Ai[(1, 1)] = di[1]
    Ai[(2, 2)] = di[2]
    Ai[(3, 3)] = di[3]
    Ai[(1, 0)] = -di[1] @ off[(1, 0)] @ di[0]
    Ai[(2, 1)] = -di[2] @ off[(2, 1)] @ di[1]
    Ai[(3, 2)] = -di[3] @ off[(3, 2)] @ di[2]
    Ai[(2, 0)] = -di[2] @ (off[(2, 0)] @ di[0] + off[(2, 1)] @ Ai[(1, 0)])
    Ai[(3, 1)] = -di[3] @ (off[(3, 1)] @ di[1] + off[(3, 2)] @ Ai[(2, 1)])
    Ai[(3, 0)] = -di[3] @ (off[(3, 0)] @ di[0] + off[(3, 1)] @ Ai[(1, 0)]
                           + off[(3, 2)] @ Ai[(2, 0)])
    # 写回: Ai[(i,j)] 形状 [B,NT,H,BC,BC] -> 需要 [B,NT,BC,H,BC] 放到 [B,NT,BT,H,BT] 切片
    for (i, j) in [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2),
                   (3, 0), (3, 1), (3, 2), (3, 3)]:
        a_i, a_j = i * BC, j * BC
        # Ai[(i,j)]: [B,NT,H,BC,BC] -> [B,NT,BC,H,BC]
        blk = Ai[(i, j)].permute(0, 1, 3, 2, 4)  # [B,NT,BC,H,BC]
        Akk_inv.view(B, NT, BT, H, BT)[:, :, a_i:a_i + BC, :, a_j:a_j + BC] = blk
    return Aqk[:, :T].contiguous(), Akk_inv[:, :T].contiguous()


# ═══════════════════════════════════════════════════════════════════════════
# triton kernel：融合单 kernel（head-merged）+ 重复平方截断逆（npow=3）
#
# 每个 CTA 处理 1 个 (chunk, head-group)，循环 HM 个 head：
#   Phase 1: 全 chunk [BT,BT] 矩阵 Mkk / Mqk（K 维单 tile），Mqk 写 Aqk（block-strict-lower，
#            对角 Aqk 来自 K2），Mkk 取 strict-lower 得 L；
#   Phase 2: (I-L)^{-1} 用重复平方链 (I-L)(I+L2)(I+L4)(I+L8)，npow=3 -> 6 个 dot。
# 对角 16x16 块由内部 Mkk 直接给出（与 K2 的 Akkd 数学一致），因此无需读 Akkd。
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit(do_not_specialize=["T", "TP"])
def _inter_solve_kernel(
    q, k, g, beta, Aqk, Akk_out,
    scale, T, TP, H: tl.constexpr, K: tl.constexpr,
    BT: tl.constexpr, BC: tl.constexpr, HM: tl.constexpr, NP: tl.constexpr,
    NS: tl.constexpr,
):
    """融合 K3：grid=(cdiv(T,BT), cdiv(B*H,HM))。batch 由 i_hg 解码。

    NP 截断级数: 1 级=2 dot, 2 级=4 dot, 3 级=6 dot。默认 NP=2（6.04ms, 3.3e-04）。
    """
    i_tc, i_hg = tl.program_id(0), tl.program_id(1)
    if i_tc * BT >= T:
        return
    n_hg = H // HM
    i_b = i_hg // n_hg
    hg0 = i_hg % n_hg
    i_tc0 = i_tc * BT
    s_k = H * K
    s_A = H * BT
    s_beta = H
    b_off = i_b * T
    b_offA = i_b * TP  # 缓冲已按 TP=NT*BT 补齐, store 需用 TP 步长
    r = tl.arange(0, BT)
    c = tl.arange(0, BT)
    m_t = (i_tc0 + r) < T
    m_blk = (r // BC)[:, None] > (r // BC)[None, :]
    b_I = tl.where(r[:, None] == c[None, :], 1.0, 0.0)
    o_k = tl.arange(0, K)
    for hh in tl.range(HM, num_stages=NS):
        i_h = hg0 * HM + hh
        qs = q + b_off * s_k + i_tc0 * s_k + i_h * K
        ks = k + b_off * s_k + i_tc0 * s_k + i_h * K
        gs = g + b_off * s_k + i_tc0 * s_k + i_h * K
        b_beta = tl.load(beta + b_off * s_beta + (i_tc0 + r) * s_beta + i_h,
                         mask=m_t, other=0.0).to(tl.float32)
        b_q = tl.load(qs + r[:, None] * s_k + o_k[None, :],
                      mask=m_t[:, None], other=0.0).to(tl.float32)
        b_k = tl.load(ks + r[:, None] * s_k + o_k[None, :],
                      mask=m_t[:, None], other=0.0).to(tl.float32)
        b_g = tl.load(gs + r[:, None] * s_k + o_k[None, :],
                      mask=m_t[:, None], other=0.0).to(tl.float32)
        b_eg = tl.math.exp2(b_g)
        b_Ke = b_k * tl.math.exp2(-b_g)
        b_Mkk = tl.dot(b_k * b_eg * b_beta[:, None], tl.trans(b_Ke))
        b_Mqk = tl.dot(b_q * b_eg * scale, tl.trans(b_Ke))
        As = Aqk + b_offA * s_A + i_tc0 * s_A + i_h * BT
        tl.store(As + r[:, None] * s_A + c[None, :],
                 tl.where(m_blk, b_Mqk, 0.0).to(Aqk.dtype.element_ty))
        b_L = tl.where(r[:, None] > c[None, :], b_Mkk, 0.0)
        b_inv = b_I - b_L
        b_pow = b_L
        b_pow = tl.dot(b_pow, b_pow)
        b_inv = tl.dot(b_inv, b_I + b_pow)
        if NP >= 2:
            b_pow = tl.dot(b_pow, b_pow)
            b_inv = tl.dot(b_inv, b_I + b_pow)
        if NP >= 3:
            b_pow = tl.dot(b_pow, b_pow)
            b_inv = tl.dot(b_inv, b_I + b_pow)
        if NP >= 4:
            b_pow = tl.dot(b_pow, b_pow)
            b_inv = tl.dot(b_inv, b_I + b_pow)
        if NP >= 5:
            b_pow = tl.dot(b_pow, b_pow)
            b_inv = tl.dot(b_inv, b_I + b_pow)
        Os = Akk_out + b_offA * s_A + i_tc0 * s_A + i_h * BT
        tl.store(Os + r[:, None] * s_A + c[None, :],
                 b_inv.to(Akk_out.dtype.element_ty))


def turbo_inter_solve_triton(
    q, k, g, beta, Akkd, scale,
    Aqk=None, Akk_out=None,
    chunk_size=_BT, sub_chunk_size=_BC,
):
    """融合单 kernel 版（head-merged, 重复平方截断逆）。返回 (Aqk, Akk_inv)。

    忽略 Akkd：对角 16×16 块由内部 Mkk 直接给出（与 K2 的 Akkd 数学一致）。
    K 需为 2 幂（单 tile tl.arange）；HM 见 K3_HM（默认 16，H 不整除时回退 1）。
    """
    B, T, H, K = q.shape
    BT, BC = chunk_size, sub_chunk_size
    NT = _cdiv(T, BT)
    TP = NT * BT
    dev = q.device
    # fp16 形态(B): 输入统一 fp16、缓冲 fp16, kernel 内 fp32 cube(见 EXPERIMENT_FP16.md:
    # fp16 dot 在该 kernel 的深依赖链上触发 triton-ascend 3.2.1 运行期 507015)
    q = q.to(torch.float16)
    k = k.to(torch.float16)
    g = g.to(torch.float16)
    beta = beta.to(torch.float16)
    # 缓冲按 NT*BT 补齐 + torch.empty: kernel 做无掩码全量写回（掩码处写 0.0），
    # 消除 tail 行掩码 store 的标量开销与每次调用的 2×402MB memset（ZerosLike）。
    if Aqk is None:
        Aqk = torch.empty(B, TP, H, BT, device=dev, dtype=torch.float16)
    if Akk_out is None:
        Akk_out = torch.empty(B, TP, H, BT, device=dev, dtype=torch.float16)
    HM = _HM if H % _HM == 0 else 1
    grid = (NT, B * (H // HM))
    _inter_solve_kernel[grid](
        q, k, g, beta, Aqk, Akk_out, float(scale), T, TP,
        H=H, K=K, BT=BT, BC=BC, HM=HM, NP=_NP, NS=_NUM_STAGES,
        num_warps=_NUM_WARPS,
    )
    torch.npu.synchronize()
    return Aqk[:, :T], Akk_out[:, :T]
