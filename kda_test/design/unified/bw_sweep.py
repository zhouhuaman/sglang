#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""严格 A/B：同一地址集、两种 CTA 顺序 + num_warps 对 strided-tile 带宽的影响。

张量 x,y 视为 [NR, S]，S=H*K=12288（[B,T,H,K] 的 T 维行距）。每个 tile
[R=64, C=128]（1 head）。两种顺序覆盖**完全相同**的地址集：
  * ROW-MAJOR (ORDER=0): 连续 pid 先走列块 c_idx → 跨 CTA 地址连续
  * COL-MAJOR (ORDER=1): 连续 pid 先走行块 r_idx → 跨 CTA 地址跳 R*S
同时扫 num_warps，检验 K1 用 nw=1 是否为带宽瓶颈。
"""
import argparse
import time

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

torch.npu.set_device("npu:0")


@triton.jit
def _tile_copy(x, y, R: tl.constexpr, C: tl.constexpr, S: tl.constexpr,
               NGR: tl.constexpr, NC: tl.constexpr, ORDER: tl.constexpr):
    pid = tl.program_id(0)
    if ORDER == 0:  # row-major: pid_r = pid // NC, pid_c = pid % NC
        pid_r = pid // NC
        pid_c = pid % NC
    else:           # col-major: pid_r = pid % NGR, pid_c = pid // NGR
        pid_r = pid % NGR
        pid_c = pid // NGR
    o_r = tl.arange(0, R)
    o_c = tl.arange(0, C)
    base = pid_r * (R * S) + pid_c * C
    src = x + base + o_r[:, None] * S + o_c[None, :]
    dst = y + base + o_r[:, None] * S + o_c[None, :]
    tl.store(dst, tl.load(src))


def bench(fn, R=30, W=6):
    for _ in range(W):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(R):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / R


def main():
    B, T, H, K = 1, 16384, 96, 128
    S = H * K
    n = B * T * H * K
    x = torch.randn(n, dtype=torch.float32, device="npu:0")
    y = torch.empty(n, dtype=torch.float32, device="npu:0")
    gb = 2 * n * 4 / (1024 ** 3)
    print(f"# x=[{n}]  S={S}  tile [R,C]=[64,128]  read+write {gb:.2f}GB\n", flush=True)
    print(f"{'order':>10} {'nw':>3}  {'us':>8}  {'GB/s':>6}", flush=True)

    R, C = 64, 128
    NGR, NC = n // (R * S), S // C   # 行块数=256, 列块数=96
    grid = (NGR * NC,)
    for order in (0, 1):
        for nw in (1, 2, 4):
            dt = bench(lambda: _tile_copy[grid](
                x, y, R=R, C=C, S=S, NGR=NGR, NC=NC, ORDER=order, num_warps=nw))
            print(f"{'row-major' if order==0 else 'col-major':>10} {nw:>3}  "
                  f"{dt*1e6:8.1f}  {gb/dt:6.0f}", flush=True)


if __name__ == "__main__":
    main()
