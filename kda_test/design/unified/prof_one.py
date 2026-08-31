#!/usr/bin/env python3
"""只跑指定 triton kernel（输入从 k235_inputs.pt 加载），供 msprof 隔离指标。
用法: python3 prof_one.py K2|K3|K5"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch_npu  # noqa: F401

import bench  # noqa: E402

torch.npu.set_device("npu:0")
kid = sys.argv[1]
d = torch.load("k235_inputs.pt", map_location="npu:0")
args = tuple(t.to("npu:0") if torch.is_tensor(t) else t for t in d[kid])
ki = {kid: {"triton_args": args, "triton_kwargs": d[kid + "_kw"]}}
for _ in range(3):
    bench._call_kernel(kid, "triton", ki)
torch.npu.synchronize()
print(kid, "done")
