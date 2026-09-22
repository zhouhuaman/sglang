#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""KDA Kernel-5（Delta Rule H）独立实现：纯 torch + torch_npu + triton。

本模块是 ``python/sglang/kernels/ops/attention/fla/chunk_delta_h.py`` 中
``chunk_gated_delta_rule_fwd_h`` 的功能等价物，但只依赖 ``torch`` /
``torch_npu`` / ``triton``，**不 import 任何 sglang 代码**，因此可被独立
验证目录使用。

只保留**固定长度 + USE_GK + USE_EXP2 + INPLACE_UPDATE + SAVE_NEW_VALUE +
USE_INITIAL_STATE** 路径（K==V），删除上游的 VARLEN / USE_G / USE_EXP2=False
等分支。这与本项目（Kimi-Linear Delta Attention）实际调用路径一致：上游
``chunk_kda()`` 调用本 kernel 时 ``cu_seqlens=None``、``use_exp2=True``、
``initial_state`` 非空、``save_new_value=True``。

计算内容（与上游 kernel 完全一致）:

    State h ∈ [V, K] 跨 chunk 递推:
        for c in 0..NT-1:
            h_snapshot[c] = h                       # 保存快照到 h[B, NT, H, V, K]
            v_new = u - w @ h^T                     # Delta Rule: 残差 = 原值 - 历史预测
            v_new_save[c] = v_new                   # 保存到 v_new[B, T, H, V]
            h = h * exp2(gk_last)                   # per-channel 衰减（log2 空间）
            h += k^T @ v_new                        # 外积累加

    Epilogue: 把最终 state 写回 initial_state（in-place）。

Triton kernel 为 tiled 向量化实现:

    * grid = ``(cdiv(V, BV), N * H)``，BV = 32（env ``SGLANG_GDN_CHUNK_H_BV``）；
    * 每个 program (CTA) 处理一个 ``((batch, head), V-tile)``，加载 BT=64 个
      token 的 kg/w/u 切片，K 维按 64 分 4 个 tile 展开（K≤256）；
    * 状态寄存器 ``b_h1..b_h4`` 形状 ``[BV, 64]`` fp32；K=64 时只用 b_h1，
      K=128 用 b_h1/b_h2，依此类推；
    * 由于 triton-ascend 编译器对 ``block_ptr`` store 到 ``(V, K)`` 形状的
      源寄存器有损坏 bug，快照与最终写回统一用行步长 K 的 2D 手动指针
      ``tl.store``（``_store_h_full``，支持任意 K；勿用 flat reshape store，
      CANN 9.1 的 expand_shape 会报 "collapsed dim size" 错误）；
    * 去掉了 K/V 列的 boundary_check（恒入界），保留 T 维行边界；
    * gk_last 加载: K=64/128 时 mask 恒真直接去掉，K>128 用 fp32 比较。
"""

import os

import torch
import torch_npu  # noqa: F401  (必须在创建任何 npu 张量之前 import)
import triton
import triton.language as tl


_BT = 64                              # chunk 大小
_BV = int(os.getenv("SGLANG_GDN_CHUNK_H_BV", "32"))    # V 维 tile 大小
_NUM_WARPS = int(os.getenv("SGLANG_GDN_CHUNK_H_NUM_WARPS", "4"))
_NUM_STAGES = int(os.getenv("SGLANG_GDN_CHUNK_H_NUM_STAGES", "3"))  # 第五轮: tl.range 流水线 NS=3 最优 9.11ms


def _cdiv(a: int, b: int) -> int:
    """向上取整的整数除法。"""
    return -(a // -b)


# ═══════════════════════════════════════════════════════════════════════════
# torch CPU 参考（ground truth）—— 逐 chunk 递推，整体 [V, K] 计算
# ═══════════════════════════════════════════════════════════════════════════

def delta_rule_h_ref(
    k, w, u, gk, initial_state, initial_state_indices,
    chunk_size=_BT,
):
    """纯 torch CPU 参考实现（可在任意 device 上运行，典型为 CPU）。

    与上游 ``chunk_gated_delta_rule_fwd_kernel_h_blockdim64`` 数学一致:
      * 不做 BV 分块（整体 [V, K] 计算），简化逻辑；
      * USE_GK + USE_EXP2 + INPLACE_UPDATE + SAVE_NEW_VALUE + USE_INITIAL_STATE；
      * K == V 约束（上游 kernel 同样要求 K==V）。

    参数:
        k (kg):  [B, T, H, K] fp32  衰减后的 key（k * beta * exp2(gk_last - gk)）
        w:       [B, T, H, K] fp32  衰减后的 w
        u:       [B, T, H, V] fp32  原始 value (= Aqk @ (v * beta))
        gk:      [B, T, H, K] fp32  per-channel gate（log2 空间，已 cumsum + scale）
        initial_state: [N, H, V, K] fp32  初始状态（in-place 更新为最终状态）
        initial_state_indices: [B] int32  每个 batch 条目指向 initial_state 的索引

    返回:
        h:     [B, NT, H, V, K] fp32  每 chunk 起始状态快照
        v_new: [B, T, H, V] fp32       Delta Rule 残差 value
        initial_state: [N, H, V, K]    （已被 in-place 更新为最终状态）
    """
    B, T, H, K = k.shape
    V = u.shape[-1]
    assert K == V, f"delta_rule_h_ref requires K==V, got K={K}, V={V}"
    BT = int(chunk_size)
    NT = _cdiv(T, BT)

    kf = k.float()
    wf = w.float()
    uf = u.float()
    gkf = gk.float()
    state0 = initial_state.float().clone()
    indices = initial_state_indices.to(torch.int64).cpu()

    h = torch.zeros(B, NT, H, V, K, dtype=torch.float32)
    v_new = torch.zeros(B, T, H, V, dtype=torch.float32)

    for b in range(B):
        idx = int(indices[b].item())
        for h_idx in range(H):
            state = state0[idx, h_idx].clone()   # [V, K]
            for c in range(NT):
                tc = c * BT
                tc_end = min(T, tc + BT)
                BT_act = tc_end - tc
                # ① 保存快照（chunk 起始状态）
                h[b, c, h_idx] = state
                # ② Delta Rule: v_new = u - w @ state^T
                w_chunk = wf[b, tc:tc_end, h_idx]          # [BT_act, K]
                u_chunk = uf[b, tc:tc_end, h_idx]          # [BT_act, V]
                k_chunk = kf[b, tc:tc_end, h_idx]          # [BT_act, K]
                v_c = u_chunk - w_chunk @ state.T          # [BT_act, V]
                v_new[b, tc:tc_end, h_idx] = v_c
                # ③ per-channel gate 衰减: state *= exp2(gk_last)
                last = tc_end - 1
                gk_last = gkf[b, last, h_idx]              # [K]
                state = state * torch.exp2(gk_last[None, :])   # [V, K]
                # ④ 外积累加: state += k^T @ v_c  (=[K, BT] @ [BT, V] -> [K, V], 再转置)
                state = state + v_c.T @ k_chunk            # [V, K] += [V, BT] @ [BT, K]
            # 写回最终 state
            state0[idx, h_idx] = state

    # in-place 更新 initial_state
    initial_state.copy_(state0.to(initial_state.dtype))
    return h, v_new


# ═══════════════════════════════════════════════════════════════════════════
# torch_npu 元算子版本（精度/性能基准）—— 逐 chunk 串行，每 chunk 内用 matmul
# ═══════════════════════════════════════════════════════════════════════════

def delta_rule_h_torch(
    k, w, u, gk, initial_state, initial_state_indices,
    chunk_size=_BT,
):
    """torch_npu 元算子版本: 与 ``delta_rule_h_ref`` 数学一致, 在 NPU 上运行。

    chunk 间有依赖（state 跨 chunk 传递），无法完全批量化；每个 chunk 内用
    ``torch.matmul`` / ``torch.exp2`` / 广播乘法（在 NPU 上各为一个 kernel）。
    作为精度基准（与 ref 一致）与性能基准（多 kernel 拼接 vs triton 单 kernel）。

    参数与 ``delta_rule_h_ref`` 相同，所有张量须在 NPU 上。
    返回 (h, v_new)；initial_state 被 in-place 更新。
    """
    B, T, H, K = k.shape
    V = u.shape[-1]
    assert K == V, f"delta_rule_h_torch requires K==V, got K={K}, V={V}"
    BT = int(chunk_size)
    NT = _cdiv(T, BT)

    kf = k.to(torch.float32)
    wf = w.to(torch.float32)
    uf = u.to(torch.float32)
    gkf = gk.to(torch.float32)
    dev = k.device
    state0 = initial_state.to(torch.float32).clone()
    indices = initial_state_indices.to(torch.int64)

    h = torch.zeros(B, NT, H, V, K, dtype=torch.float32, device=dev)
    v_new = torch.zeros(B, T, H, V, dtype=torch.float32, device=dev)

    for b in range(B):
        idx = int(indices[b].item())
        for h_idx in range(H):
            state = state0[idx, h_idx].clone()   # [V, K]
            for c in range(NT):
                tc = c * BT
                tc_end = min(T, tc + BT)
                # ① 快照
                h[b, c, h_idx] = state
                # ② Delta Rule
                w_chunk = wf[b, tc:tc_end, h_idx]
                u_chunk = uf[b, tc:tc_end, h_idx]
                k_chunk = kf[b, tc:tc_end, h_idx]
                v_c = u_chunk - w_chunk @ state.T
                v_new[b, tc:tc_end, h_idx] = v_c
                # ③ per-channel 衰减
                last = tc_end - 1
                gk_last = gkf[b, last, h_idx]              # [K]
                state = state * torch.exp2(gk_last[None, :])
                # ④ 外积更新
                state = state + v_c.T @ k_chunk
            state0[idx, h_idx] = state

    initial_state.copy_(state0.to(initial_state.dtype))
    return h, v_new


# ═══════════════════════════════════════════════════════════════════════════
# triton kernel：固定长度 + USE_GK + USE_EXP2 子集（与上游一致）
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def _exp2(x):
    """log2 空间的 exp2; 上游用 tl.math.exp2（非 fast_expf 路径）。"""
    return tl.math.exp2(x)


@triton.jit(do_not_specialize=["T"])
def _delta_rule_h_kernel(
    k, v, w, v_new, gk, h, initial_state, initial_state_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    NS: tl.constexpr,
):
    """Delta Rule H triton kernel（固定长度 + USE_GK + USE_EXP2 子集）。

    Grid = (cdiv(V, BV), N * H)；每个 CTA 处理 ((batch, head), V-tile i_v)。
    递推: snapshot -> Delta Rule -> per-channel decay -> 外积更新 -> 写回。

    第二轮优化（BT 不变）: K 维不再按 64 分 tile，整 K 作单 tile（b_h [BV, K]）。
    K=128 时每 chunk 从 4 dot 降到 2 dot（隔离实验: 4 dot 占 7ms/10ms），
    K=64 时本就 2 dot，语义完全一致。目标 case 10.1ms -> 9.36ms。

    snapshot/epilogue 的 store 统一走 _store_h_full 的通用 2D 手动指针
    （任意 K，含 K=64：勿用 flat reshape store —— triton-ascend MLIR 的
    expand_shape 在 CANN 9.1 下报错，见 _store_h_full docstring）。
    """
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    # 固定长度: 每 batch 长度相同 T
    bos, eos = i_n * T, i_n * T + T
    NT = tl.cdiv(T, BT)
    boh = i_n * NT

    # 整 K 单 tile state 寄存器 [BV, K]（K≤256；基准 case K∈{64,128}）
    b_h = tl.zeros([BV, K], dtype=tl.float32)
    offs_v = i_v * BV + tl.arange(0, BV)
    offs_k = tl.arange(0, K)

    # 偏移到本 (batch, head)
    stride_v = H * V
    stride_h = H * V * K
    stride_k = Hg * K
    stride_w = H * K

    h += (boh * H + i_h) * V * K
    v += (bos * H + i_h) * V
    k += (bos * Hg + i_h // (H // Hg)) * K
    w += (bos * H + i_h) * K
    v_new += (bos * H + i_h) * V

    index = tl.load(initial_state_indices + i_n).to(tl.int32)
    h0 = initial_state + index * stride_h
    ht = initial_state + index * stride_h
    h0 = h0 + i_h * V * K
    ht = ht + i_h * V * K

    # 加载初始状态 — 无 boundary_check（V≥BV, K≥64 恒入界）
    p_h0 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, K), (1, 0))
    b_h += tl.load(p_h0).to(tl.float32)

    # 主循环: 逐 chunk 递推（tl.range + num_stages 软件流水线，预取下一 chunk 的
    # w/u/k 加载以掩盖序列链点积延迟；本包内即该最终收敛版）
    for i_t in tl.range(NT, num_stages=NS):
        # ① 保存快照
        _store_h_full(h, i_t * stride_h, i_v * BV, K, BV, b_h)

        # ② Delta Rule: b_v = u - w @ h^T（整 K 单 dot）
        p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, K), (1, 0))
        b_w = tl.load(p_w)
        b_v = tl.dot(b_w, tl.trans(b_h).to(b_w.dtype))

        # v (u) 加载: 只保留 T 维 boundary_check=(0,)
        p_v = tl.make_block_ptr(v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0,)) - b_v

        # 保存 v_new
        p_v_new = tl.make_block_ptr(v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        tl.store(p_v_new, b_v.to(p_v_new.dtype.element_ty), boundary_check=(0,))

        # ③ per-channel gate 衰减 (USE_GK + USE_EXP2)，整 K 无 mask（K≥64 恒入界）
        last_idx = min((i_t + 1) * BT, T) - 1
        b_gk_last = tl.load(gk + (bos + last_idx) * H * K + i_h * K + offs_k)
        b_h *= _exp2(b_gk_last)[None, :]
        b_v = b_v.to(k.dtype.element_ty)

        # ④ 外积更新: b_h += b_v^T @ k（[BV,BT]@[BT,K]）。
        # 原写法 b_h += trans(dot(k, b_v)) 在 CANN 9.1 触发 hivm-plan-memory
        # "Unsupported op for finding the root alloc" → ub overflow 误报；
        # 行主 [BT,K] 加载 + 输入侧转置可正常编译（数学等价: trans(b_v)@b_k）。
        # 实测: 此形态 11.5ms（vs 基线 9.15ms, +25%）；trans(dot) 加 where 打断
        # 可编译但 21.8ms；b_ht 转置形态 22.3ms —— 均为当前最优可用方案。
        p_k2 = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, K), (1, 0))
        b_k2 = tl.load(p_k2)
        b_h += tl.dot(tl.trans(b_v), b_k2)

    # Epilogue: 写回最终 state
    _store_h_full(ht, 0, i_v * BV, K, BV, b_h)


@triton.jit
def _store_h_full(base, chunk_offset, v_start, K, BV, b_h):
    """通用 2D 手动指针 store（支持任意 K，含 K=64）。

    b_h 为整 K 单 tile [BV, K]。
    注意: 不用 flat reshape store —— triton-ascend MLIR 的 expand_shape 在
    CANN 9.1 下报 "collapsed dim size 2048 must equal 4096"（K=64 实测，
    原始版 + zeros workaround 同样失败）；K=64 走本 2D 路径（行步长恰为 K，
    每行 64 元素连续），与 K≠64 一致。
    """
    offs_v = v_start + tl.arange(0, BV)
    offs_k = tl.arange(0, K)
    ptr = base + chunk_offset + offs_v[:, None] * K + offs_k[None, :]
    tl.store(ptr, b_h.to(base.dtype.element_ty))


def delta_rule_h_triton(
    k, w, u, gk, initial_state, initial_state_indices,
    chunk_size=_BT, BV=None, num_warps=None, num_stages=None,
):
    """triton kernel 版: 在 NPU 上运行，返回 (h, v_new)，initial_state 被 in-place 更新。

    参数:
        k (kg):  [B, T, H, K]  衰减后的 key (kg = k * beta * exp2(gk_last - gk))
        w:       [B, T, H, K]  衰减后的 w
        u:       [B, T, H, V]  原始 value (= Aqk @ (v * beta))
        gk:      [B, T, H, K]  per-channel gate (log2 空间, 已 cumsum + scale)
        initial_state: [N, H, V, K]  初始状态（in-place 更新为最终状态）
        initial_state_indices: [B] int32  每个 batch 指向 initial_state 的索引
        chunk_size: chunk 大小（默认 64，与上游一致）
        BV: V 维 tile 大小（默认 V，整 V 驻留一个 CTA；目标 case 比 BV=32 快 4x）
        num_warps: 每 CTA warp 数（默认 env SGLANG_GDN_CHUNK_H_NUM_WARPS=4）
        num_stages: pipeline stage 数（默认 env SGLANG_GDN_CHUNK_H_NUM_STAGES=2）

    返回:
        h:     [B, NT, H, V, K]  每 chunk 起始状态快照
        v_new: [B, T, H, V]      Delta Rule 残差 value
        initial_state: [N, H, V, K]  （已被 in-place 更新）

    纯 CPU / 无 NPU 环境下不可用（请用 ``delta_rule_h_ref``）。
    """
    B, T, Hg, K = k.shape
    V = u.shape[-1]
    H = u.shape[-2]
    assert K == V, f"delta_rule_h_triton requires K==V, got K={K}, V={V}"
    assert K <= 256, "current kernel does not support head dimension larger than 256."
    BT = int(chunk_size)
    NT = _cdiv(T, BT)
    if BV is None or BV <= 0:
        # 目标大 case (V=128, 256 chunks): BV=V (整 V 驻留一个 CTA) 比 BV=32 快 4x
        # （BV=32 -> 40ms, BV=64 -> 20ms, BV=128 -> 10ms, 实测 BV=V 单调最优且正确）
        BV = V
    if num_warps is None:
        num_warps = _NUM_WARPS
    if num_stages is None:
        num_stages = _NUM_STAGES

    h = k.new_empty(B, NT, H, V, K)
    v_new = torch.empty_like(u)

    grid = (_cdiv(V, BV), B * H)
    _delta_rule_h_kernel[grid](
        k, u, w, v_new, gk, h, initial_state, initial_state_indices,
        T,
        H=H, Hg=Hg, K=K, V=V, BT=BT, BV=BV, NS=num_stages,
        num_warps=num_warps, num_stages=num_stages,
    )
    torch.npu.synchronize()
    return h, v_new
