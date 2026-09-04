#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""KDA Kernel-4 (Recompute W/U) 独立实现：纯 torch + triton。

本模块是 ``python/sglang/kernels/ops/attention/fla/kda.py`` 中
``recompute_w_u_fwd`` 的功能等价物，但只依赖 ``torch`` / ``triton``，
**不 import 任何 sglang 代码**，可被独立验证目录使用。

计算内容（与上游 kernel 一致）::

    给定 chunk 内的 KKT 逆矩阵 ``A`` (= Akk_inv, [BT, BT])、Key/Value
    张量与 chunk 局部 gate cumsum ``gk``，重新计算解耦后的 w/u/kg:

      1. w = A @ (k * beta * exp2(gk))           [BT, K]    Key 解耦表示
      2. u = A @ (v * beta)                       [BT, V]    Value 解耦表示
      3. kg = k * exp2(gk_last - gk)              [BT, K]    时间对齐 Key
         (gk_last = chunk 内最后一个有效 token 的 gk 值)

    w/u 已去掉 chunk 内的因果依赖，可参与跨 chunk 递推; kg 把 chunk 内
    每个 token 的 Key 对齐到 chunk 末尾时间戳, 供后续 chunk_gla_fwd_o_gk 使用。

Triton kernel 为分块 matmul 实现:

    * grid = ``(NT, ceil(B*H/HM))``，每 CTA 处理 ``HM`` 个 head 的
      ``(chunk, batch)``（头合并，grid 24576 -> 1536 CTA，实测 6.54 -> 4.78 ms）;
    * 公共加载: ``beta[BT]`` 与 ``A_inv[BT, BT]`` (每个 head 一次, 留在寄存器);
    * V 维度循环: 每 ``BV=32`` 切一块, ``v' = v*beta`` → ``u = A @ v'`` (tl.dot);
    * K 维度循环: 每 ``BK=32`` 切一块, ``k' = k*beta*exp2(gk)`` → ``w = A @ k'``,
      同时复用 k_tile/gk_tile 计算 ``kg = k * exp2(gk_last - gk)``。

尾部 (partial) chunk 处理
-------------------------
最后一个 chunk 的行数可能不足 BT，K/V 维也可能出现最后一个 tile 不足 BK/BV。
kernel 用 ``tl.make_block_ptr`` 的 ``boundary_check`` 处理: 越界位置 load 为 0,
因此 ``A @ (k*beta*exp2(gk))`` 中越界行的零乘积自然为 0，不会污染有效行。

关键差异（独立子集化）
---------------------
* 只做固定长度 (B,T,H,K,V)，不做 VARLEN / ``cu_seqlens`` / ``chunk_indices``;
* ``STORE_KG`` 由上层 driver 根据 ``gk is not None`` 决定 (与上游一致);
* ``DOT_PRECISION="tf32"`` 与上游一致 (triton-ascend 上等价于 ieee 精度);
* ``HM`` 头合并: driver 在 ``H % HM == 0`` 时启用 (HM=16), 否则 HM=1 等价旧行为。
"""

import torch
import torch_npu  # noqa: F401  (必须在创建任何 npu 张量之前 import)

import triton
import triton.language as tl

# chunk 大小 (BT) 与 K/V 维 tile 大小 (BK/BV)，与上游 autotune 默认配置一致。
_DEFAULT_BT = 64
_DEFAULT_BK = 32
_DEFAULT_BV = 32
# 头合并: 每 CTA 处理的 head 数 (H=96 时 16 -> grid 24576/1536 CTA)。
_DEFAULT_HM = 16


def _cdiv(a: int, b: int) -> int:
    """向上取整的整数除法。"""
    return -(a // -b)


# ═══════════════════════════════════════════════════════════════════════════
# torch CPU 参考（ground truth）
# ═══════════════════════════════════════════════════════════════════════════

def recompute_w_u_ref(
    k, v, beta, A, gk=None,
    chunk_size=_DEFAULT_BT,
):
    """纯 torch CPU 参考实现（逐 chunk 循环, ground truth）。

    参数:
        k:  [B, T, H, K]
        v:  [B, T, H, V]
        beta: [B, T, H]
        A:  [B, T, H, BT]  (Akk_inv, 每 chunk 内 [BT, BT] 的下三角逆)
        gk: [B, T, H, K] 或 None  (chunk 局部 gate cumsum, log2 空间)

    返回:
        w:  [B, T, H, K]
        u:  [B, T, H, V]
        kg: [B, T, H, K] 或 None
    """
    B_, T_, H_, K_ = k.shape
    V_ = v.shape[-1]
    BT = A.shape[-1]
    NT = _cdiv(T_, BT)

    kf = k.float()
    vf = v.float()
    betaf = beta.float()
    Af = A.float()
    gkf = gk.float() if gk is not None else None

    # 尾 chunk 不满 BT 时补 0 到 NT*BT, 保证 chunk 切片等长
    pad = NT * BT - T_
    if pad:
        zf = torch.zeros
        kf = torch.cat([kf, zf(B_, pad, H_, K_, dtype=torch.float32)], dim=1)
        vf = torch.cat([vf, zf(B_, pad, H_, V_, dtype=torch.float32)], dim=1)
        betaf = torch.cat([betaf, zf(B_, pad, H_, dtype=torch.float32)], dim=1)
        Af = torch.cat([Af, zf(B_, pad, H_, BT, dtype=torch.float32)], dim=1)
        if gkf is not None:
            gkf = torch.cat([gkf, zf(B_, pad, H_, K_, dtype=torch.float32)], dim=1)

    w = torch.zeros(B_, NT * BT, H_, K_, dtype=torch.float32)
    u = torch.zeros(B_, NT * BT, H_, V_, dtype=torch.float32)
    kg = torch.zeros(B_, NT * BT, H_, K_, dtype=torch.float32) if gkf is not None else None

    for b in range(B_):
        for h in range(H_):
            for c in range(NT):
                tc = c * BT
                A_chunk = Af[b, tc:tc + BT, h]                # [BT, BT]
                kb = kf[b, tc:tc + BT, h]                    # [BT, K]
                vb = vf[b, tc:tc + BT, h]                    # [BT, V]
                bb = betaf[b, tc:tc + BT, h]                 # [BT]

                # u = A @ (v * beta)
                vb_scaled = vb * bb.unsqueeze(-1)            # [BT, V]
                u[b, tc:tc + BT, h] = A_chunk @ vb_scaled

                if gkf is not None:
                    gkc = gkf[b, tc:tc + BT, h]              # [BT, K]
                    # w = A @ (k * beta * exp2(gk))
                    kb_scaled = kb * bb.unsqueeze(-1) * torch.exp2(gkc)
                    w[b, tc:tc + BT, h] = A_chunk @ kb_scaled
                    # kg = k * exp2(gk_last - gk)
                    # gk_last = chunk 内最后一个有效 token 的 gk (与上游 kernel 一致)
                    last = min(tc + BT, T_) - 1
                    gk_last = gkf[b, last, h]                # [K]
                    kg[b, tc:tc + BT, h] = kb * torch.exp2(gk_last.unsqueeze(0) - gkc)
                else:
                    # 无 gk: 退化为 w = A @ (k * beta), 不输出 kg
                    kb_scaled = kb * bb.unsqueeze(-1)
                    w[b, tc:tc + BT, h] = A_chunk @ kb_scaled

    w = w[:, :T_].contiguous()
    u = u[:, :T_].contiguous()
    if kg is not None:
        kg = kg[:, :T_].contiguous()
    return w, u, kg


# ═══════════════════════════════════════════════════════════════════════════
# torch_npu 元算子版本（精度/性能基准）——按 chunk 批量化, 避免 Python 逐 chunk 循环
# ═══════════════════════════════════════════════════════════════════════════

def recompute_w_u_torch(
    k, v, beta, A, gk=None,
    chunk_size=_DEFAULT_BT,
):
    """torch 元算子版本: 与 ``recompute_w_u_ref`` 数学一致, 但按 chunk 批量化。

    利用 reshape + 批量 matmul 完成 ``A @ (k * beta * exp2(gk))`` 与
    ``A @ (v * beta)`` 的计算, 减少 Python 循环开销, 作为**性能基准**
    供与 triton kernel 做加速比对比。

    返回值与 ref 相同: (w, u, kg | None), 全部 fp32。
    """
    B_, T_, H_, K_ = k.shape
    V_ = v.shape[-1]
    BT = A.shape[-1]
    NT = _cdiv(T_, BT)
    dev = k.device

    kf = k.float()
    vf = v.float()
    betaf = beta.float()
    Af = A.float()
    gkf = gk.float() if gk is not None else None

    pad = NT * BT - T_
    if pad:
        z = torch.zeros
        kf = torch.cat([kf, z(B_, pad, H_, K_, dtype=torch.float32, device=dev)], dim=1)
        vf = torch.cat([vf, z(B_, pad, H_, V_, dtype=torch.float32, device=dev)], dim=1)
        betaf = torch.cat([betaf, z(B_, pad, H_, dtype=torch.float32, device=dev)], dim=1)
        Af = torch.cat([Af, z(B_, pad, H_, BT, dtype=torch.float32, device=dev)], dim=1)
        if gkf is not None:
            gkf = torch.cat([gkf, z(B_, pad, H_, K_, dtype=torch.float32, device=dev)], dim=1)

    # [B, NT, BT, H, *] reshape: chunk 内 BT 行, 方便批量 matmul
    Kb = kf.reshape(B_, NT, BT, H_, K_)
    Vb = vf.reshape(B_, NT, BT, H_, V_)
    Bb = betaf.reshape(B_, NT, BT, H_)
    Ab = Af.reshape(B_, NT, BT, H_, BT)

    # u = A @ (v * beta): [B,NT,BT,H,V] = [B,NT,BT,H,BT] @ [B,NT,BT,H,V] * [B,NT,BT,H,1]
    vb = Vb * Bb.unsqueeze(-1)
    # 把 H 移到 matmul 的 batch 维: [B, NT, H, BT, V] = [B, NT, H, BT, BT] @ [B, NT, H, BT, V]
    u = (Ab.permute(0, 1, 3, 2, 4) @ vb.permute(0, 1, 3, 2, 4)).permute(0, 1, 3, 2, 4)

    if gkf is not None:
        Gb = gkf.reshape(B_, NT, BT, H_, K_)
        # w = A @ (k * beta * exp2(gk))
        kb_scaled = Kb * Bb.unsqueeze(-1) * torch.exp2(Gb)
        w = (Ab.permute(0, 1, 3, 2, 4) @ kb_scaled.permute(0, 1, 3, 2, 4)).permute(0, 1, 3, 2, 4)
        # kg = k * exp2(gk_last - gk): gk_last = chunk 内最后一个有效 token 的 gk
        # 尾 chunk 不满 BT 时, 最后一个有效 token = Gb[:, :, (T - 1) % BT, :, :]
        last_in_chunk = torch.full((B_, NT, 1, H_, 1), min(T_, BT) - 1, dtype=torch.long, device=dev)
        if T_ % BT:
            last_in_chunk[:, -1, 0, :, 0] = (T_ - 1) % BT
        gk_last = torch.gather(Gb, dim=2, index=last_in_chunk.expand(B_, NT, 1, H_, K_))  # [B, NT, 1, H, K]
        kg = Kb * torch.exp2(gk_last - Gb)
    else:
        kb_scaled = Kb * Bb.unsqueeze(-1)
        w = (Ab.permute(0, 1, 3, 2, 4) @ kb_scaled.permute(0, 1, 3, 2, 4)).permute(0, 1, 3, 2, 4)
        kg = None

    # 展平回 [B, T, H, *]
    w = w.reshape(B_, NT * BT, H_, K_)[:, :T_].contiguous()
    u = u.reshape(B_, NT * BT, H_, V_)[:, :T_].contiguous()
    if kg is not None:
        kg = kg.reshape(B_, NT * BT, H_, K_)[:, :T_].contiguous()
    return w, u, kg


# ═══════════════════════════════════════════════════════════════════════════
# triton kernel：Manual pointer arithmetic + 头合并 (HM loop)
# 优化策略: 消除 boundary_check 标量退避 + K/V 维无 mask + num_warps=4
#           + HM 头合并 (grid 24576 -> 1536 CTA, 实测 6.54 -> 4.78 ms)
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit(do_not_specialize=["T"])
def _recompute_w_u_kernel(
    k, kg, v, beta, w, u, A, gk,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    STORE_KG: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    T_FULL: tl.constexpr,
    HM: tl.constexpr,
):
    """1 CTA / (chunk, HM heads). BK=K, BV=V, single tile. Manual pointers + head loop."""
    i_t, i_hg = tl.program_id(0), tl.program_id(1)
    hg0 = i_hg * HM
    i_b = hg0 // H
    i_h0 = hg0 % H
    bos = i_b * T
    base = i_t * BT

    # Stride
    s_k = H * K
    s_v = H * V
    s_beta = H
    s_A = H * BT

    # Row/col indices
    o_bt = tl.arange(0, BT)
    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)
    if T_FULL:
        m_t = None
    else:
        m_t = (base + o_bt).to(tl.float32) < T

    for hh in range(HM):
        i_h = i_h0 + hh
        off_bh = (bos * H + i_h)

        # Base pointers for this (batch, head)
        p_k = k + off_bh * K
        p_v = v + off_bh * V
        p_beta = beta + off_bh
        p_A = A + off_bh * BT
        p_w = w + off_bh * K
        p_u = u + off_bh * V
        if STORE_KG:
            p_kg = kg + off_bh * K
            p_gk = gk + off_bh * K

        # ── A_inv [BT, BT] — manual pointer ──
        b_A = tl.load(p_A + (base + o_bt[:, None]) * s_A + o_bt[None, :]).to(tl.float32)
        # ── beta [BT] ──
        b_b = tl.load(p_beta + (base + o_bt) * s_beta).to(tl.float32)

        # ── V 维单 tile: u = A @ (v * beta) ──
        if T_FULL:
            b_v = tl.load(p_v + (base + o_bt[:, None]) * s_v + o_v[None, :]).to(tl.float32)
        else:
            b_v = tl.load(p_v + (base + o_bt[:, None]) * s_v + o_v[None, :],
                          mask=m_t[:, None] & (o_v[None, :] < V), other=0.0).to(tl.float32)
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, input_precision=DOT_PRECISION)
        if T_FULL:
            tl.store(p_u + (base + o_bt[:, None]) * s_v + o_v[None, :], b_u.to(u.dtype.element_ty))
        else:
            tl.store(p_u + (base + o_bt[:, None]) * s_v + o_v[None, :],
                     b_u.to(u.dtype.element_ty), mask=m_t[:, None] & (o_v[None, :] < V))

        # ── K 维单 tile: w (=A @ (k * beta * exp2(gk))) (+ kg) ──
        if T_FULL:
            b_k = tl.load(p_k + (base + o_bt[:, None]) * s_k + o_k[None, :]).to(tl.float32)
        else:
            b_k = tl.load(p_k + (base + o_bt[:, None]) * s_k + o_k[None, :],
                          mask=m_t[:, None] & (o_k[None, :] < K), other=0.0).to(tl.float32)
        b_kb = b_k * b_b[:, None]

        if STORE_KG:
            if T_FULL:
                b_gk = tl.load(p_gk + (base + o_bt[:, None]) * s_k + o_k[None, :]).to(tl.float32)
            else:
                b_gk = tl.load(p_gk + (base + o_bt[:, None]) * s_k + o_k[None, :],
                               mask=m_t[:, None] & (o_k[None, :] < K), other=0.0).to(tl.float32)
            b_kb = b_kb * tl.math.exp2(b_gk)

            last_idx = tl.minimum(base + BT, T) - 1
            b_gn = tl.load(p_gk + last_idx * s_k + o_k).to(tl.float32)
            b_kg = b_k * tl.math.exp2(b_gn[None, :] - b_gk)

            if T_FULL:
                tl.store(p_kg + (base + o_bt[:, None]) * s_k + o_k[None, :], b_kg.to(kg.dtype.element_ty))
            else:
                tl.store(p_kg + (base + o_bt[:, None]) * s_k + o_k[None, :],
                         b_kg.to(kg.dtype.element_ty), mask=m_t[:, None] & (o_k[None, :] < K))

        b_w = tl.dot(b_A, b_kb.to(b_k.dtype), input_precision=DOT_PRECISION)
        if T_FULL:
            tl.store(p_w + (base + o_bt[:, None]) * s_k + o_k[None, :], b_w.to(w.dtype.element_ty))
        else:
            tl.store(p_w + (base + o_bt[:, None]) * s_k + o_k[None, :],
                     b_w.to(w.dtype.element_ty), mask=m_t[:, None] & (o_k[None, :] < K))


def recompute_w_u_triton(
    k, v, beta, A, gk=None,
    chunk_size=_DEFAULT_BT,
    num_warps=4,
):
    """triton kernel 版 (Route A + 头合并): 返回 (w, u, kg | None)。输入张量已在 NPU。

    优化策略: BK=K, BV=V 单 tile 直通 + T_FULL constexpr 分派 + num_warps=4
              + HM 头合并 (H%16==0 时 16 head/CTA, grid 24576 -> 1536 CTA)。

    参数与上游 ``recompute_w_u_fwd`` 兼容:
        k:  [B, T, H, K]  fp32 / bf16
        v:  [B, T, H, V]  fp32 / bf16
        beta: [B, T, H]   fp32 / bf16
        A:  [B, T, H, BT] Akk_inv (fp32 / bf16, kernel 内部转 fp32 计算)
        gk: [B, T, H, K] | None  (chunk 局部 gate cumsum, log2 空间)
        chunk_size: chunk 大小 (默认 64, 必须等于 A.shape[-1])

    返回:
        w:  [B, T, H, K]  与 k 同 dtype
        u:  [B, T, H, V]  与 v 同 dtype
        kg: [B, T, H, K] | None  (与 k 同 dtype)
    """
    B, T, H, K = k.shape
    V = v.shape[-1]
    BT = A.shape[-1]
    assert BT == chunk_size, f"A.shape[-1]={BT} != chunk_size={chunk_size}"

    # 统一转 fp32 并搬上 NPU (不修改调用方张量)
    k = k.to(torch.float32).to("npu")
    v = v.to(torch.float32).to("npu")
    beta = beta.to(torch.float32).to("npu")
    A = A.to(torch.float32).to("npu")
    has_gk = gk is not None
    if has_gk:
        gk = gk.to(torch.float32).to("npu")
    else:
        gk = k  # STORE_KG=False 时该指针不会被读取, 传 dummy

    w = torch.empty_like(k)
    u = torch.empty_like(v)
    kg = torch.empty_like(k) if has_gk else None
    kg_ptr = kg if kg is not None else k  # dummy

    NT = _cdiv(T, BT)
    HM = _DEFAULT_HM if (H % _DEFAULT_HM == 0) else 1
    grid = (NT, _cdiv(B * H, HM))
    T_FULL = (T % BT == 0)

    _recompute_w_u_kernel[grid](
        k=k,
        kg=kg_ptr,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        gk=gk,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        STORE_KG=has_gk,
        DOT_PRECISION="tf32",
        T_FULL=T_FULL,
        HM=HM,
        num_warps=num_warps,
    )
    torch.npu.synchronize()
    return w, u, kg
