#!/usr/bin/env python3
"""K6 fp32 vs bf16 输入 A/B（同一进程、交替测，控制共享设备噪声）。

只测 triton kernel 的 mean 耗时 + triton(bf16) vs torch(fp32) 的 max_diff。
bf16 路径 = 把 K6 五个输入 cast 到 bf16 再喂现有 kernel（dot 若触发 cube 会有
明显加速；traffic 减半但 scalar-bound 未必受益）。
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch_npu  # noqa: F401

import bench  # noqa: E402

torch.npu.set_device("npu:0")
B, T, H, K, V = 1, 16384, 96, 128, 128
torch.manual_seed(20260815)
base = bench._gen_case_inputs(B, T, H, K)
ki = bench._derive_inputs(base, "npu:0")
a = ki["K6"]["triton_args"]
q, v, g, A, h, scale = a


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


def run_fp32():
    return bench._call_kernel("K6", "triton", ki)


def run_bf16():
    args = (q.bfloat16(), v.bfloat16(), g.bfloat16(), A.bfloat16(),
            h.bfloat16(), scale)
    kk = {"K6": {"triton_args": args, "triton_kwargs": {"chunk_size": bench._BT}}}
    return bench._call_kernel("K6", "triton", kk)


# 预热并校验 bf16 精度
o32 = run_fp32()
ob = run_bf16()
print(f"max_diff(torch fp32 vs triton bf16) = {bench._max_diff(o32, ob):.2e}", flush=True)

t32, tb = [], []
for _ in range(2):  # 交替 2 轮控制噪声
    t32.append(mean(run_fp32))
    tb.append(mean(run_bf16))
print(f"fp32: {min(t32):8.1f}us   bf16: {min(tb):8.1f}us   "
      f"speedup(baseline/bf16)={min(t32)/min(tb):.2f}x", flush=True)
