#!/usr/bin/env python3
"""派生 K2/K3/K5 的 triton 输入并统一存盘，供 prof_one.py 单独 profile。"""
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
out = {}
for kid in ("K2", "K3", "K5"):
    a = ki[kid]["triton_args"]
    out[kid] = [t.cpu() if torch.is_tensor(t) else t for t in a]
    out[kid + "_kw"] = ki[kid]["triton_kwargs"]
torch.save(out, "k235_inputs.pt")
for kid in ("K2", "K3", "K5"):
    shapes = [tuple(t.shape) if torch.is_tensor(t) else t for t in out[kid]]
    print(kid, shapes, out[kid + "_kw"], flush=True)
