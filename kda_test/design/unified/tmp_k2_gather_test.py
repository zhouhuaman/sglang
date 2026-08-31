#!/usr/bin/env python3
"""最小复现 v2: tl.gather 在 triton-ascend 3.2.1 上的行为。

变体 A: src = load 的 [64,64], idx = load 的          (已确认: 正确)
变体 B: src = tl.dot 输出 [64,64], idx = load 的       (dot 输出布局?)
变体 C: src = tl.dot 输出, idx = 计算出的 (非 load)    (最贴近 hm3)

对照: torch out_ref = src.gather(1, idx)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "token_parallel", "src"))

import torch  # noqa: E402
import torch_npu  # noqa: F401
import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def _gather_load_src(a, b, idx, out, M: tl.constexpr, N: tl.constexpr,
                     W: tl.constexpr):
    """src = a @ b (dot), idx 从 GMEM load。"""
    src = tl.dot(a, tl.trans(b))
    i = tl.load(idx + tl.arange(0, M)[:, None] * W + tl.arange(0, W)[None, :])
    g = tl.gather(src, i, axis=1)
    tl.store(out + tl.arange(0, M)[:, None] * W + tl.arange(0, W)[None, :], g)


@triton.jit
def _gather_dot_computed(a, b, idx_unused, out, M: tl.constexpr, N: tl.constexpr,
                         W: tl.constexpr, BC: tl.constexpr):
    """src = a @ b (dot), idx 从 arange 计算 (与 hm3 的 col_idx 完全一致)。"""
    src = tl.dot(a, tl.trans(b))
    o_r = tl.arange(0, M)
    o_cc = tl.arange(0, W)
    i = (o_r // BC)[:, None] * BC + o_cc[None, :]
    g = tl.gather(src, i, axis=1)
    tl.store(out + tl.arange(0, M)[:, None] * W + tl.arange(0, W)[None, :], g)


def main():
    torch.npu.set_device("npu:0")
    M, N, W, BC = 64, 64, 16, 16
    torch.manual_seed(20260815)
    a = torch.randn(M, N, device="npu:0")
    b = torch.randn(M, N, device="npu:0")
    src_ref = (a @ b.t())
    r = torch.arange(M, device="npu:0")
    idx = (r[:, None] // BC * BC + torch.arange(W, device="npu:0")[None, :]).to(
        torch.int32)
    ref = src_ref.gather(1, idx.long())

    out = torch.empty(M, W, device="npu:0")
    _gather_load_src[(1,)](a, b, idx, out, M=M, N=N, W=W)
    torch.npu.synchronize()
    print(f"B(dot src, load idx) max_diff = {(out.cpu() - ref.cpu()).abs().max().item():.3e}")

    out2 = torch.empty(M, W, device="npu:0")
    _gather_dot_computed[(1,)](a, b, idx, out2, M=M, N=N, W=W, BC=BC)
    torch.npu.synchronize()
    print(f"C(dot src, computed idx) max_diff = {(out2.cpu() - ref.cpu()).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
