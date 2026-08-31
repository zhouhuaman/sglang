#!/usr/bin/env python3
"""通用单 kernel 隔离 profile：任意 K1..K6，输入经 derive_prefix 派生。
用法: python3 prof_any.py K4
（msprof --ai-core=on --output=prof_any_$kid --application="python3 prof_any.py K4"）"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch_npu  # noqa: F401

import bench  # noqa: E402
import time_kernels  # noqa: E402

torch.npu.set_device("npu:0")
kid = sys.argv[1]
B, T, H, K = 1, 16384, 96, 128
torch.manual_seed(20260815)
base = bench._gen_case_inputs(B, T, H, K)
ki = time_kernels.derive_prefix(kid, base, "npu:0")
for _ in range(3):
    bench._call_kernel(kid, "triton", ki)
torch.npu.synchronize()
print(kid, "done")
