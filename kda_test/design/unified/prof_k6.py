#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""单 kernel (K6) 最小 profiling 脚本：派生输入 + 跑几次 k6_triton。

供 msprof --ai-core 捕获，定位 K6 瓶颈（memory vs cube vs vector）。
用法（容器内）:
    msprof --ai-core=on --output=./prof_k6 python3 prof_k6.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch_npu  # noqa: F401

import bench  # noqa: E402

torch.npu.set_device("npu:0")

B, T, H, K, V = 1, 16384, 96, 128, 128
torch.manual_seed(20260815)
base = bench._gen_case_inputs(B, T, H, K)
ki = bench._derive_inputs(base, "npu:0")  # 完整 K1->K6 前缀链

# warmup + 若干次计时（msprof 只关心 kernel task）
for _ in range(3):
    bench._call_kernel("K6", "triton", ki)
torch.npu.synchronize()
for _ in range(5):
    bench._call_kernel("K6", "triton", ki)
torch.npu.synchronize()
print("K6 done")
