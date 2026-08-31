#!/usr/bin/env python3
"""K4 exp2 削减 A/B：baseline(2×[BT,K] exp2) vs 重写(1×exp2 + 倒数 + [K]exp2)。
同进程交替测 + 精度 vs torch。"""
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
import time_kernels  # noqa: E402

torch.npu.set_device("npu:0")
B, T, H, K, V = 1, 16384, 96, 128, 128
torch.manual_seed(20260815)
base = bench._gen_case_inputs(B, T, H, K)
ki = time_kernels.derive_prefix("K4", base, "npu:0")
a = ki["K4"]["triton_args"]
k, v, beta, A, gk = a
BT = 64
TP = (T // BT) * BT
o_w = torch.empty(B, TP, H, K, dtype=torch.float32, device=k.device)
o_u = torch.empty(B, TP, H, V, dtype=torch.float32, device=k.device)
o_kg = torch.empty(B, TP, H, K, dtype=torch.float32, device=k.device)


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


@triton.jit(do_not_specialize=["T"])
def _k4_base(k, kg, v, beta, w, u, A, gk, T,
             H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
             BT: tl.constexpr, HM: tl.constexpr):
    i_t, i_hg = tl.program_id(0), tl.program_id(1)
    hg0 = i_hg * HM
    i_b = hg0 // H
    i_h0 = hg0 % H
    bos = i_b * T
    base = i_t * BT
    s_k = H * K
    s_v = H * V
    s_beta = H
    s_A = H * BT
    o_bt = tl.arange(0, BT)
    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)
    for hh in range(HM):
        i_h = i_h0 + hh
        off_bh = bos * H + i_h
        p_A = A + off_bh * BT
        p_beta = beta + off_bh
        b_A = tl.load(p_A + (base + o_bt[:, None]) * s_A + o_bt[None, :]).to(tl.float32)
        b_b = tl.load(p_beta + (base + o_bt) * s_beta).to(tl.float32)
        b_v = tl.load(v + off_bh * V + (base + o_bt[:, None]) * s_v + o_v[None, :]).to(tl.float32)
        b_vb = (b_v * b_b[:, None])
        b_u = tl.dot(b_A, b_vb)
        tl.store(u + off_bh * V + (base + o_bt[:, None]) * s_v + o_v[None, :], b_u.to(u.dtype.element_ty))
        b_k = tl.load(k + off_bh * K + (base + o_bt[:, None]) * s_k + o_k[None, :]).to(tl.float32)
        b_kb = b_k * b_b[:, None]
        b_gk = tl.load(gk + off_bh * K + (base + o_bt[:, None]) * s_k + o_k[None, :]).to(tl.float32)
        b_kb = b_kb * tl.math.exp2(b_gk)
        last_idx = tl.minimum(base + BT, T) - 1
        b_gn = tl.load(gk + off_bh * K + last_idx * s_k + o_k).to(tl.float32)
        b_kg = b_k * tl.math.exp2(b_gn[None, :] - b_gk)
        tl.store(kg + off_bh * K + (base + o_bt[:, None]) * s_k + o_k[None, :], b_kg.to(kg.dtype.element_ty))
        b_w = tl.dot(b_A, b_kb)
        tl.store(w + off_bh * K + (base + o_bt[:, None]) * s_k + o_k[None, :], b_w.to(w.dtype.element_ty))


@triton.jit(do_not_specialize=["T"])
def _k4_opt(k, kg, v, beta, w, u, A, gk, T,
            H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
            BT: tl.constexpr, HM: tl.constexpr):
    i_t, i_hg = tl.program_id(0), tl.program_id(1)
    hg0 = i_hg * HM
    i_b = hg0 // H
    i_h0 = hg0 % H
    bos = i_b * T
    base = i_t * BT
    s_k = H * K
    s_v = H * V
    s_beta = H
    s_A = H * BT
    o_bt = tl.arange(0, BT)
    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)
    for hh in range(HM):
        i_h = i_h0 + hh
        off_bh = bos * H + i_h
        p_A = A + off_bh * BT
        p_beta = beta + off_bh
        b_A = tl.load(p_A + (base + o_bt[:, None]) * s_A + o_bt[None, :]).to(tl.float32)
        b_b = tl.load(p_beta + (base + o_bt) * s_beta).to(tl.float32)
        b_v = tl.load(v + off_bh * V + (base + o_bt[:, None]) * s_v + o_v[None, :]).to(tl.float32)
        b_vb = (b_v * b_b[:, None])
        b_u = tl.dot(b_A, b_vb)
        tl.store(u + off_bh * V + (base + o_bt[:, None]) * s_v + o_v[None, :], b_u.to(u.dtype.element_ty))
        b_k = tl.load(k + off_bh * K + (base + o_bt[:, None]) * s_k + o_k[None, :]).to(tl.float32)
        b_gk = tl.load(gk + off_bh * K + (base + o_bt[:, None]) * s_k + o_k[None, :]).to(tl.float32)
        # 重写: 只算一次 exp2(gk)，kg 用 exp2(gk_last)/exp2(gk) 共享
        b_eg = tl.math.exp2(b_gk)                     # [BT,K] 唯一的大 exp2
        b_kb = b_k * b_b[:, None] * b_eg
        last_idx = tl.minimum(base + BT, T) - 1
        b_gn = tl.load(gk + off_bh * K + last_idx * s_k + o_k).to(tl.float32)
        b_egn = tl.math.exp2(b_gn)                    # [K] 小 exp2
        b_kg = b_k * (b_egn[None, :] / b_eg)          # exp2(a-b)=exp2(a)/exp2(b)
        tl.store(kg + off_bh * K + (base + o_bt[:, None]) * s_k + o_k[None, :], b_kg.to(kg.dtype.element_ty))
        b_w = tl.dot(b_A, b_kb)
        tl.store(w + off_bh * K + (base + o_bt[:, None]) * s_k + o_k[None, :], b_w.to(w.dtype.element_ty))


def make(kernel, HM):
    grid = (T // BT, (B * H) // HM)
    def run():
        kernel[grid](k, o_kg, v, beta, o_w, o_u, A, gk, T,
                     H=H, K=K, V=V, BT=BT, HM=HM, num_warps=4)
        torch.npu.synchronize()
        return o_w, o_u, o_kg
    return run


r = {}
r["base"] = make(_k4_base, 16)
r["opt"] = make(_k4_opt, 16)

ref_w, ref_u, ref_kg = r["base"]()
# 精度: torch 参考
torch_out = bench._call_kernel("K4", "torch", ki)
for name, run in r.items():
    if name == "base":
        continue
    w, u, kg = run()
    d_w = bench._max_diff(torch_out[0], w[:, :T])
    d_u = bench._max_diff(torch_out[1], u[:, :T])
    d_kg = bench._max_diff(torch_out[2], kg[:, :T])
    print(f"{name}: max_diff vs torch  w={d_w:.2e} u={d_u:.2e} kg={d_kg:.2e}", flush=True)

times = {k: [] for k in r}
for _ in range(3):
    for name, run in r.items():
        times[name].append(mean(run))
for name, ts in times.items():
    print(f"{name:<6} {min(ts):8.1f}us  (all: {[f'{x:.0f}' for x in ts]})", flush=True)
