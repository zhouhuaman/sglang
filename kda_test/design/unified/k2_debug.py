#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""K2 (token_parallel) 目标 case 独立精度定位脚本。

目标 case: D_KV128_H96_T16384 (B=1, T=16384, H=96, K=V=128)。
输入分布与 unified/bench.py 完全一致（固定种子 20260815）:

    q,k = L2-normalize(randn); g = randn*0.1; beta = sigmoid(rand); scale = 1/sqrt(K)

三路实现对比（torch 元算子 = ground truth）:
    * torch : token_parallel_torch                  （精度基准）
    * hm3   : token_parallel_triton 当前默认路径（driver 已切回 hm2 + driver
              torch.gather 收拢；hm3 kernel 内 tl.gather 在 triton-ascend 3.2.1
              上对 dot 输出数值错误, 见 kernel 内 DEPRECATED 注释）
    * hm2   : _token_parallel_kernel_hm2 + driver torch.gather 收拢（同路径,
              手动构造 A/B 对照）

定位输出: 整体 max_diff / top-N head / top-N chunk / 位置热图 [BT,BT]/[BT,BC]，
用于快速判定错误来源（tl.gather? head 索引? chunk 边界? sub-chunk?）。

用法:
    python3 k2_debug.py                       # 默认: hm3 路径 + 定位分析
    python3 k2_debug.py --impl hm2            # 只跑 hm2 路径
    python3 k2_debug.py --impl both           # 双路径 A/B
    python3 k2_debug.py --T 1024 --H 8        # 小 shape 冒烟 (快速迭代)
    python3 k2_debug.py --dump k2_inputs.pt   # 导出输入供外部复现
    python3 k2_debug.py --no-locate           # 只报整体 max_diff
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DESIGN = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(DESIGN, "token_parallel", "src"))

import torch  # noqa: E402
import torch_npu  # noqa: F401  (必须先于 npu 张量 import)
import triton  # noqa: E402

import token_parallel_kernel as tp  # noqa: E402

BT, BC = 64, 16
SEED = 20260815
DEV = "npu:0"


# ─── 输入生成（与 bench.py::_gen_case_inputs 分布一致）───────────────────────


def gen_inputs(B, T, H, K, device=DEV):
    """固定种子生成 q/k/g/beta/scale（fp32, 搬 NPU）。"""
    torch.manual_seed(SEED)
    q = torch.nn.functional.normalize(
        torch.randn(B, T, H, K, dtype=torch.float32), dim=-1)
    k = torch.nn.functional.normalize(
        torch.randn(B, T, H, K, dtype=torch.float32), dim=-1)
    g = torch.randn(B, T, H, K, dtype=torch.float32) * 0.1
    beta = torch.rand(B, T, H, dtype=torch.float32).sigmoid()
    scale = 1.0 / (float(K) ** 0.5)
    return (q.to(device), k.to(device), g.to(device), beta.to(device), scale)


# ─── 三路实现 ────────────────────────────────────────────────────────────────


def run_torch(q, k, g, beta, scale):
    return tp.token_parallel_torch(q, k, g, beta, scale,
                                   chunk_size=BT, sub_chunk_size=BC)


def run_hm3(q, k, g, beta, scale):
    """默认 triton 路径: token_parallel_triton（driver 现走 hm2 语义）。"""
    return tp.token_parallel_triton(q, k, g, beta, scale,
                                    chunk_size=BT, sub_chunk_size=BC)


def run_hm2(q, k, g, beta, scale):
    """hm2 路径: scratch 满宽写 + driver torch.gather 收拢（A/B 对照）。"""
    B, T, H, K = q.shape
    NT = tp._cdiv(T, BT)
    TP = NT * BT
    Aqk = torch.empty(B, TP, H, BT, device=q.device, dtype=torch.float32)
    scratch = torch.empty(B, TP, H, BT, device=q.device, dtype=torch.float32)
    HM = 16 if H % 16 == 0 else 1
    grid = (NT, B * (H // HM))
    tp._token_parallel_kernel_hm2[grid](
        q, k, g, beta, Aqk, scratch, float(scale),
        T, H=H, K=K, BT=BT, BC=BC, HM=HM, num_warps=1,
    )
    torch.npu.synchronize()
    return Aqk[:, :T], tp._gather_akk_diag(scratch, BC, T=T)


def run_hm1(q, k, g, beta, scale):
    """hm1 路径: 原 _token_parallel_kernel (1 CTA/(chunk,head)) + torch.gather。"""
    B, T, H, K = q.shape
    BK = triton.next_power_of_2(K)
    NT = tp._cdiv(T, BT)
    TP = NT * BT
    Aqk = torch.empty(B, TP, H, BT, device=q.device, dtype=torch.float32)
    scratch = torch.empty(B, TP, H, BT, device=q.device, dtype=torch.float32)
    grid = (B * NT, H)
    tp._token_parallel_kernel[grid](
        q, k, g, beta, Aqk, scratch, float(scale),
        T, H=H, K=K, BT=BT, BC=BC, BK=BK, num_warps=1,
    )
    torch.npu.synchronize()
    return Aqk[:, :T], tp._gather_akk_diag(scratch, BC, T=T)


# ─── 定位分析 ────────────────────────────────────────────────────────────────


def _max_diff(a, b):
    return (a.float().cpu() - b.float().cpu()).abs().max().item()


def _topk_flat(hm, inner, name, n=5):
    """位置热图 [BT,inner] 的 top-N 位置。"""
    vals, idx = hm.flatten().topk(n)
    out = []
    for v, ix in zip(vals.tolist(), idx.tolist()):
        r, c = divmod(ix, inner)
        out.append(f"({r},{c})={v:.3e}")
    print(f"    {name} top-{n} 位置: " + "  ".join(out))


def locate(ref, tri, name, B, T, H, inner):
    """分 head/chunk/位置输出差异热区。ref/tri: [B,T,H,inner]。"""
    d = (ref.float().cpu() - tri.float().cpu()).abs()
    NT = T // BT
    # per-head
    per_h = d.amax(dim=(0, 1, 3))                       # [H]
    vh, ih = per_h.topk(min(5, H))
    print(f"    {name} top-{len(ih)} head: " + "  ".join(
        f"h{i}={v:.3e}" for v, i in zip(vh.tolist(), ih.tolist())))
    # per-chunk
    d5 = d.reshape(B, NT, BT, H, inner)
    per_c = d5.amax(dim=(0, 2, 3, 4))                   # [NT]
    vc, ic = per_c.topk(min(5, NT))
    print(f"    {name} top-{len(ic)} chunk: " + "  ".join(
        f"c{i}={v:.3e}" for v, i in zip(vc.tolist(), ic.tolist())))
    # 位置热图（归约掉 B/chunk/head）
    hm = d5.amax(dim=(0, 3))                            # [NT, BT, inner]
    hm = hm.amax(dim=0)                                 # [BT, inner]
    _topk_flat(hm, inner, name)
    # 非零比例（>1e-6 视为有差异）
    frac = (d > 1e-6).float().mean().item()
    print(f"    {name} 差异元素比例(>1e-6): {frac * 100:.2f}%")
    return d.max().item()


def compare(ref, tri, B, T, H, tag, locate_=True):
    """对比 (Aqk, Akk) 两个输出，返回整体 max_diff。"""
    print(f"\n== {tag} ==")
    for name, r, t in (("Aqk", ref[0], tri[0]), ("Akk", ref[1], tri[1])):
        inner = r.shape[-1]
        dmax = _max_diff(r, t)
        print(f"  {name} max_diff = {dmax:.3e}")
        if locate_ and dmax > 1e-4:
            locate(r, t, name, B, T, H, inner)
    return max(_max_diff(ref[0], tri[0]), _max_diff(ref[1], tri[1]))


def bench(fn, *args, warmup=1, repeats=3):
    """简单 wall-clock 计时（仅迭代用，非 msprof 口径）。"""
    fn(*args)
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(repeats):
        fn(*args)
    torch.npu.synchronize()
    return (time.time() - t0) / repeats * 1000  # ms/调用


# ─── main ────────────────────────────────────────────────────────────────────


def main(argv=None):
    p = argparse.ArgumentParser(description="K2 token_parallel 独立精度定位")
    p.add_argument("--impl", choices=["hm3", "hm2", "hm1", "both"], default="hm3",
                   help="triton 路径（默认 hm3; both=hm3+hm2 对照）")
    p.add_argument("--B", type=int, default=1)
    p.add_argument("--T", type=int, default=16384)
    p.add_argument("--H", type=int, default=96)
    p.add_argument("--K", type=int, default=128)
    p.add_argument("--no-locate", action="store_true", help="只报整体 max_diff")
    p.add_argument("--dump", default=None, help="导出输入到 .pt 文件")
    p.add_argument("--device", default=DEV)
    a = p.parse_args(argv)

    torch.npu.set_device(a.device)
    B, T, H, K = a.B, a.T, a.H, a.K

    print(f"# K2 debug: B={B} T={T} H={H} K={K} device={a.device} "
          f"(BT={BT} BC={BC} seed={SEED})")
    q, k, g, beta, scale = gen_inputs(B, T, H, K, a.device)
    if a.dump:
        torch.save({"q": q.cpu(), "k": k.cpu(), "g": g.cpu(),
                    "beta": beta.cpu(), "scale": scale}, a.dump)
        print(f"# 输入已导出 -> {a.dump}")

    ref = run_torch(q, k, g, beta, scale)
    torch.npu.synchronize()
    print("# torch ref 完成 (Aqk/Akk 就绪)")

    impls = {"hm3": run_hm3, "hm2": run_hm2, "hm1": run_hm1}
    targets = ["hm3", "hm2"] if a.impl == "both" else [a.impl]
    worst = 0.0
    for tag in targets:
        t0 = time.time()
        tri = impls[tag](q, k, g, beta, scale)
        torch.npu.synchronize()
        dt = time.time() - t0
        md = compare(ref, tri, B, T, H, f"{tag} (triton, {dt * 1000:.0f}ms 含编译)",
                     locate_=not a.no_locate)
        worst = max(worst, md)
        ms = bench(impls[tag], q, k, g, beta, scale)
        print(f"  {tag} wall-clock ≈ {ms:.1f} ms/调用 (warmup1+repeat3, 仅迭代用)")
    print(f"\n# 结论: 最大 max_diff = {worst:.3e} "
          f"({'OK (<1e-2)' if worst < 1e-2 else 'FAIL (≥1e-2)'})")
    return 0 if worst < 1e-2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
