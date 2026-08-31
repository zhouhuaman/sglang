#!/usr/bin/env python3
"""只跑 K6 triton kernel（输入从 k6_inputs.pt 加载），隔离 K6 的 msprof 指标。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch_npu  # noqa: F401

import bench  # noqa: E402

torch.npu.set_device("npu:0")
args = torch.load("k6_inputs.pt", map_location="npu:0")
q, v_new, g, Aqk, h, scale = args
ki = {"K6": {
    "triton_args": (q, v_new, g, Aqk, h, scale),
    "triton_kwargs": {"chunk_size": bench._BT},
}}
for _ in range(3):
    bench._call_kernel("K6", "triton", ki)
torch.npu.synchronize()
print("K6-only done")
