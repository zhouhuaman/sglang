#!/usr/bin/env python3
"""派生 K6 输入并存盘，供 prof_k6b.py 只跑 K6 triton kernel。"""
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
ki = bench._derive_inputs(base, "npu:0")
a = ki["K6"]["triton_args"]
torch.save([t.cpu() if torch.is_tensor(t) else t for t in a], "k6_inputs.pt")
print("saved k6_inputs.pt:", [tuple(t.shape) if torch.is_tensor(t) else t for t in a])
