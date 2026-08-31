#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""910B2 内存带宽探测: 连续 vs 跨步 tile 访问的带宽对比。

K2/K3/K5 的 [64,128] tile 访问是跨步的（行距 H*K=12288 floats）。
对比连续块 copy 与跨步 tile copy 的实际带宽，判断布局是否是瓶颈。
"""
import time

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

torch.npu.set_device("npu:0")


@triton.jit
def _copy_kernel(x, y, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(y + offs, tl.load(x + offs))


# 模拟 K2/K3: 每 CTA 读 [BT=64, K=128] tile（行距 STRIDE=H*K=12288）→ 写 [BT,64]
@triton.jit
def _strided_tile_kernel(x, y, H, BT: tl.constexpr, K: tl.constexpr, OUT: tl.constexpr):
    pid = tl.program_id(0)
    o_r = tl.arange(0, BT)
    o_k = tl.arange(0, K)
    o_c = tl.arange(0, OUT)
    rows = o_r[:, None] * (H * K) + o_k[None, :]     # 行距 H*K
    cols = o_r[:, None] * (H * OUT) + o_c[None, :]
    tl.store(y + pid * (BT * OUT) + cols, tl.load(x + pid * (BT * K) + rows))


def bench(fn, R=20):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(R):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / R


def main():
    B, T, H, K = 1, 16384, 96, 128
    BT, OUT = 64, 64
    n_chunk = (T // BT) * H          # 256*96 = 24576 个 tile
    x = torch.randn(B, T, H, K, dtype=torch.float32, device="npu:0")
    y = torch.empty(B, T, H, OUT, dtype=torch.float32, device="npu:0")

    # 连续 copy 基准（同样字节数: 读 805MB 写 402MB）
    n_contig = B * T * H * K
    xc = torch.randn(n_contig, dtype=torch.float32, device="npu:0")
    yc = torch.empty(B * T * H * OUT, dtype=torch.float32, device="npu:0")
    BLK = 8192
    grid = (n_contig // BLK,)
    dt = bench(lambda: _copy_kernel[grid](xc, yc, BLOCK=BLK, num_warps=8))
    gb = (805 + 402) / 1024
    print(f"contig copy  (读805+写402MB)  {dt*1e6:8.1f}us  {gb/dt:.0f} GB/s", flush=True)

    # 跨步 tile copy（同字节数, K2 访问模式）
    for nw in (1, 4):
        grid = (n_chunk,)
        dt = bench(lambda: _strided_tile_kernel[grid](
            x, y, H, BT=BT, K=K, OUT=OUT, num_warps=nw))
        print(f"strided tile (读805+写402MB) nw={nw}  {dt*1e6:8.1f}us  {gb/dt:.0f} GB/s",
              flush=True)


if __name__ == "__main__":
    main()
