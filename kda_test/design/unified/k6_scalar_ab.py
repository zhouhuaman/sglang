#!/usr/bin/env python3
"""K6 标量削减 A/B：baseline(手动指针) vs head-merge vs block_ptr。

同一进程交替测 3 个变体，控制共享设备噪声；校验各变体与 baseline 的 max_diff。
用法（容器 triton-ascend-env-zhm 内）:
    bash time_run.sh --kernel K6   # 环境准备脚本仅接受 time_kernels.py；故直接:
    source set_env 后 python3 k6_scalar_ab.py
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch_npu  # noqa: F401
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

import bench  # noqa: E402

torch.npu.set_device("npu:0")
B, T, H, K, V = 1, 16384, 96, 128, 128
BT = bench._BT
torch.manual_seed(20260815)
base = bench._gen_case_inputs(B, T, H, K)
ki = bench._derive_inputs(base, "npu:0")
q, v, g, A, h, scale = ki["K6"]["triton_args"]
h_flat = h.reshape(B * (T // BT), H, V, K).contiguous()
NT = T // BT
scale_f = float(scale)


def mean(fn, W=3, R=7):
    for _ in range(W):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(R):
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    return sum(ts) / len(ts)


# ═══════════════════════════════════════════════════════════════════════
# 变体 0: baseline（= src/gla_output_kernel.py 的 chunk_gla_fwd_kernel_o）
# ═══════════════════════════════════════════════════════════════════════
@triton.jit(do_not_specialize=["T"])
def _k6_base(q, v, g, h, o, A, scale,
             T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
             BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    i_v = tl.program_id(0)
    i_t = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = tl.cdiv(T, BT)
    i_tg = i_b * NT + i_t
    bos = i_b * T
    s_q_t: tl.constexpr = H * K
    s_v_t: tl.constexpr = H * V
    s_h_v: tl.constexpr = K
    s_a_t: tl.constexpr = H * BT
    m_s = tl.arange(0, BT)[:, None].to(tl.float32) >= tl.arange(0, BT)[None, :].to(tl.float32)
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        q_offs = tl.arange(0, BT)[:, None] * s_q_t + tl.arange(0, BK)[None, :]
        q_mask = (i_t * BT + tl.arange(0, BT)[:, None] < T) & (i_k * BK + tl.arange(0, BK)[None, :] < K)
        b_q = tl.load(q + (bos * H + i_h) * K + i_t * BT * s_q_t + i_k * BK + q_offs, mask=q_mask, other=0.0)
        b_q = (b_q * scale).to(b_q.dtype)
        b_g = tl.load(g + (bos * H + i_h) * K + i_t * BT * s_q_t + i_k * BK + q_offs, mask=q_mask, other=0.0)
        b_qg = (b_q * tl.math.exp2(b_g)).to(b_q.dtype)
        h_offs = tl.arange(0, BV)[:, None] * s_h_v + tl.arange(0, BK)[None, :]
        h_mask = (i_v * BV + tl.arange(0, BV)[:, None] < V) & (i_k * BK + tl.arange(0, BK)[None, :] < K)
        b_h = tl.load(h + (i_tg * H + i_h) * V * K + i_v * BV * s_h_v + i_k * BK + h_offs, mask=h_mask, other=0.0)
        b_o += tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype))
    v_offs = tl.arange(0, BT)[:, None] * s_v_t + tl.arange(0, BV)[None, :]
    v_mask = (i_t * BT + tl.arange(0, BT)[:, None] < T) & (i_v * BV + tl.arange(0, BV)[None, :] < V)
    b_v = tl.load(v + (bos * H + i_h) * V + i_t * BT * s_v_t + i_v * BV + v_offs, mask=v_mask, other=0.0)
    A_offs = tl.arange(0, BT)[:, None] * s_a_t + tl.arange(0, BT)[None, :]
    A_mask = i_t * BT + tl.arange(0, BT)[:, None] < T
    b_A = tl.load(A + (bos * H + i_h) * BT + i_t * BT * s_a_t + A_offs, mask=A_mask, other=0.0)
    b_A = tl.where(m_s, b_A, 0.0).to(b_v.dtype)
    b_o += tl.dot(b_A, b_v)
    o_offs = tl.arange(0, BT)[:, None] * s_v_t + tl.arange(0, BV)[None, :]
    o_mask = (i_t * BT + tl.arange(0, BT)[:, None] < T) & (i_v * BV + tl.arange(0, BV)[None, :] < V)
    tl.store(o + (bos * H + i_h) * V + i_t * BT * s_v_t + i_v * BV + o_offs, b_o.to(o.dtype.element_ty), mask=o_mask)


# ═══════════════════════════════════════════════════════════════════════
# 变体 1: head-merge（HM heads/CTA，摊薄循环不变标量设置）
# ═══════════════════════════════════════════════════════════════════════
@triton.jit(do_not_specialize=["T"])
def _k6_hm(q, v, g, h, o, A, scale,
           T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
           BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
           HM: tl.constexpr, NS: tl.constexpr):
    i_v = tl.program_id(0)
    i_t = tl.program_id(1)
    i_hg = tl.program_id(2)
    NT = tl.cdiv(T, BT)
    n_hg = H // HM
    i_b = i_hg // n_hg
    hg0 = i_hg % n_hg
    i_tg = i_b * NT + i_t
    bos = i_b * T
    s_q_t: tl.constexpr = H * K
    s_v_t: tl.constexpr = H * V
    s_h_v: tl.constexpr = K
    s_a_t: tl.constexpr = H * BT
    r = tl.arange(0, BT)
    c = tl.arange(0, BT)
    m_s = r[:, None].to(tl.float32) >= c[None, :].to(tl.float32)
    r_mask = (i_t * BT + r) < T
    k_mask = tl.arange(0, BK) < K
    v_mask = (i_v * BV + tl.arange(0, BV)) < V
    for hh in tl.range(HM, num_stages=NS):
        i_h = hg0 * HM + hh
        b_o = tl.zeros([BT, BV], dtype=tl.float32)  # 每 head 独立累加器
        q_offs = r[:, None] * s_q_t + tl.arange(0, BK)[None, :]
        q_mask = r_mask[:, None] & k_mask[None, :]
        b_q = tl.load(q + (bos * H + i_h) * K + i_t * BT * s_q_t + q_offs, mask=q_mask, other=0.0)
        b_q = (b_q * scale).to(b_q.dtype)
        b_g = tl.load(g + (bos * H + i_h) * K + i_t * BT * s_q_t + q_offs, mask=q_mask, other=0.0)
        b_qg = (b_q * tl.math.exp2(b_g)).to(b_q.dtype)
        h_offs = tl.arange(0, BV)[:, None] * s_h_v + tl.arange(0, BK)[None, :]
        b_h = tl.load(h + (i_tg * H + i_h) * V * K + i_v * BV * s_h_v + h_offs,
                      mask=v_mask[:, None] & k_mask[None, :], other=0.0)
        b_o += tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype))
        v_offs = r[:, None] * s_v_t + tl.arange(0, BV)[None, :]
        b_v = tl.load(v + (bos * H + i_h) * V + i_t * BT * s_v_t + i_v * BV + v_offs,
                      mask=r_mask[:, None] & v_mask[None, :], other=0.0)
        A_offs = r[:, None] * s_a_t + c[None, :]
        b_A = tl.load(A + (bos * H + i_h) * BT + i_t * BT * s_a_t + A_offs, mask=r_mask[:, None], other=0.0)
        b_A = tl.where(m_s, b_A, 0.0).to(b_v.dtype)
        b_o += tl.dot(b_A, b_v)
        o_offs = r[:, None] * s_v_t + tl.arange(0, BV)[None, :]
        o_mask = r_mask[:, None] & v_mask[None, :]
        tl.store(o + (bos * H + i_h) * V + i_t * BT * s_v_t + i_v * BV + o_offs,
                 b_o.to(o.dtype.element_ty), mask=o_mask)


# ═══════════════════════════════════════════════════════════════════════
# 变体 2: block_ptr（降低 int64 逐元素地址生成）
# ═══════════════════════════════════════════════════════════════════════
@triton.jit(do_not_specialize=["T"])
def _k6_bp(q, v, g, h, o, A, scale,
           T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
           BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    i_v = tl.program_id(0)
    i_t = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = tl.cdiv(T, BT)
    i_tg = i_b * NT + i_t
    bos = i_b * T
    s_q_t: tl.constexpr = H * K
    s_v_t: tl.constexpr = H * V
    s_h_v: tl.constexpr = K
    s_a_t: tl.constexpr = H * BT
    m_s = tl.arange(0, BT)[:, None].to(tl.float32) >= tl.arange(0, BT)[None, :].to(tl.float32)
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + (bos * H + i_h) * K + i_t * BT * s_q_t + i_k * BK,
                                (T - i_t * BT, K - i_k * BK), (s_q_t, 1),
                                (0, 0), (BT, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
        b_q = (b_q * scale)
        p_g = tl.make_block_ptr(g + (bos * H + i_h) * K + i_t * BT * s_q_t + i_k * BK,
                                (T - i_t * BT, K - i_k * BK), (s_q_t, 1),
                                (0, 0), (BT, BK), (1, 0))
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        b_qg = (b_q * tl.math.exp2(b_g)).to(b_q.dtype)
        p_h = tl.make_block_ptr(h + (i_tg * H + i_h) * V * K + i_v * BV * s_h_v + i_k * BK,
                                (V - i_v * BV, K - i_k * BK), (s_h_v, 1),
                                (0, 0), (BV, BK), (1, 0))
        b_h = tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)
        b_o += tl.dot(b_qg, tl.trans(b_h))
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * V + i_t * BT * s_v_t + i_v * BV,
                            (T - i_t * BT, V - i_v * BV), (s_v_t, 1),
                            (0, 0), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT + i_t * BT * s_a_t,
                            (T - i_t * BT, BT), (s_a_t, 1),
                            (0, 0), (BT, BT), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0,)).to(tl.float32)
    b_A = tl.where(m_s, b_A, 0.0).to(b_v.dtype)
    b_o += tl.dot(b_A, b_v)
    p_o = tl.make_block_ptr(o + (bos * H + i_h) * V + i_t * BT * s_v_t + i_v * BV,
                            (T - i_t * BT, V - i_v * BV), (s_v_t, 1),
                            (0, 0), (BT, BV), (1, 0))
    tl.store(p_o, b_o.to(o.dtype.element_ty), boundary_check=(0, 1))


def make_runner(kernel, grid, kwargs):
    o = torch.empty(B, T, H, V, dtype=torch.float32, device=q.device)

    def run():
        kernel[grid](q, v, g, h_flat, o, A, scale_f, T,
                     H=H, K=K, V=V, BT=BT, **kwargs)
        torch.npu.synchronize()
        return o
    return run


runners = {}
grid_base = (V // 128, NT, B * H)  # BV=128, B*H=96
runners["base"] = make_runner(_k6_base, grid_base, {"BK": 128, "BV": 128})
runners["hm16"] = make_runner(_k6_hm, (1, NT, B * H // 16), {"BK": 128, "BV": 128, "HM": 16, "NS": 1})
runners["hm16_ns2"] = make_runner(_k6_hm, (1, NT, B * H // 16), {"BK": 128, "BV": 128, "HM": 16, "NS": 2})
runners["hm16_ns3"] = make_runner(_k6_hm, (1, NT, B * H // 16), {"BK": 128, "BV": 128, "HM": 16, "NS": 3})
runners["hm8"] = make_runner(_k6_hm, (1, NT, B * H // 8), {"BK": 128, "BV": 128, "HM": 8, "NS": 1})
runners["hm32"] = make_runner(_k6_hm, (1, NT, B * H // 32), {"BK": 128, "BV": 128, "HM": 32, "NS": 1})
runners["bp"] = make_runner(_k6_bp, grid_base, {"BK": 128, "BV": 128})

# 精度：各变体 vs baseline
ref = runners["base"]()
ok = {"base": True}
for name, run in runners.items():
    if name == "base":
        continue
    try:
        out = run()
        d = bench._max_diff(ref, out)
        ok[name] = True
        print(f"  max_diff(base vs {name}) = {d:.2e}", flush=True)
    except Exception as e:
        ok[name] = False
        print(f"  {name} FAILED: {type(e).__name__}: {str(e)[:120]}", flush=True)

# 交替测时（跳过编译失败的变体）
times = {k: [] for k in runners if ok.get(k)}
for _ in range(3):
    for name, run in runners.items():
        if not ok.get(name):
            continue
        times[name].append(mean(run))
for name, ts in times.items():
    print(f"{name:<8} {min(ts):8.1f}us  (all: {[f'{x:.0f}' for x in ts]})", flush=True)
