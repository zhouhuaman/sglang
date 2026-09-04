#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""K3 inter_solve 面试题测试脚本（本目录自包含）。

你只需读懂/改动的文件是本目录的 `inter_solve_kernel.py`（含 torch 参考 + triton
kernel）。本脚本负责：造输入 → 正确性(triton vs 同文件 torch 参考) + 性能复现。

用法（在容器内，先 `source ../env.sh`）:
    # ① 目标 case 正确性门槛（评分项，输出 PASS/FAIL 与 max_diff）
    python3 test.py
    # ② 目标 case 性能复现 —— 一条 msprof 指令（测的就是目标 case）+ 解析：
    msprof --output=./prof_k3 --application="python3 test.py --perf --repeats 7 --warmup 3" \\
        && python3 test.py --report ./prof_k3
    #    --perf 只循环 triton（供 msprof 采集）；均值以 op_summary Task Duration(us) 为准。
    # 选看: python3 test.py --selftest   # 边界多 shape 快速自检（迭代期用，非评分项）

正确性判定: 每个输出 triton-vs-torch max_diff < 1e-2 即 OK（K3 目标 case 基线约 1e-7）。
性能口径: msprof `Task Duration(us)` 每调用均值。目标 case 官方隔离基线约
14.4–14.9ms/调用（±10%）；优化目标是让这个数字下降且精度不劣于 1e-2。
"""

import argparse
import csv
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import torch  # noqa: E402
import torch_npu  # noqa: F401  (先于 torch 设备使用)
import inter_solve_kernel as K  # noqa: E402

TARGET = dict(B=1, T=16384, H=96, K=128, V=128)
MAX_DIFF = 1e-2
TRITON_NAME = "_inter_solve_kernel"
BT = 64
BC = 16
OP = "k3"


def _cdiv(a, b):
    return -(-a // b)


def _diag_akk(k, g, beta, chunk_size=BT, sub_chunk_size=BC):
    """向量化 Kernel-2 对角线 Akk 块 [B,T,H,BC] —— K3 的 Akkd 输入。

    每个 sub-chunk 的 16×16 严格下三角 gated dot：行=sub-chunk 内 token，
    列=sub-chunk 内相对偏移 j-i_ts。与上游 token_parallel 数学一致；K3 的
    triton kernel 内部按同式重算（忽略传入 Akkd），故 Akkd 必须由同一份
    q/k/g/beta 生成才能让 torch 参考与 triton 对齐。**不要改动本函数**。
    """
    B, T, H, Kk = k.shape
    dev = k.device
    NC = chunk_size // sub_chunk_size
    NT = _cdiv(T, chunk_size)
    Tp = NT * chunk_size

    def pad(x):
        if Tp == T:
            return x
        extra = torch.zeros(B, Tp - T, H, *x.shape[3:], dtype=x.dtype, device=dev)
        return torch.cat([x, extra], dim=1)

    kk, gg, bb = (pad(x) for x in (k, g, beta))

    def blk4(x):
        return x.reshape(B, NT, chunk_size, H, *x.shape[3:]).reshape(
            B, NT, NC, sub_chunk_size, H, *x.shape[3:])

    K2 = blk4(kk).reshape(B * NT * NC, sub_chunk_size, H, Kk)
    G2 = blk4(gg).reshape(B * NT * NC, sub_chunk_size, H, Kk)
    B2 = blk4(bb).reshape(B * NT * NC, sub_chunk_size, H)
    wk = K2 * B2.unsqueeze(-1)
    ex = torch.exp2(G2.unsqueeze(2) - G2.unsqueeze(1))
    M = (wk.unsqueeze(2) * K2.unsqueeze(1) * ex).sum(-1)      # [S,row,col,H]
    r = torch.arange(sub_chunk_size, dtype=torch.float32, device=dev)
    low = torch.tril(r[:, None] >= r[None, :], diagonal=-1).to(M.dtype)
    M = M * low.view(1, sub_chunk_size, sub_chunk_size, 1)
    out = M.reshape(B, NT, NC, sub_chunk_size, sub_chunk_size, H).permute(
        0, 1, 2, 3, 5, 4).reshape(B, Tp, H, sub_chunk_size)
    return out[:, :T].contiguous()


def _base_inputs(B, T, H, Kk, device, seed):
    torch.manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(B, T, H, Kk), dim=-1).to(device)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, Kk), dim=-1).to(device)
    g = (torch.randn(B, T, H, Kk) * 0.5 - 2.0).to(device)
    beta = torch.rand(B, T, H).sigmoid().to(device)
    return q, k, g, beta


def _build(B, T, H, Kk, device, seed):
    q, k, g, beta = _base_inputs(B, T, H, Kk, device, seed)
    return dict(q=q, k=k, g=g, beta=beta,
                Akkd=_diag_akk(k, g, beta), scale=1.0 / (Kk ** 0.5))


def _run_torch(i):
    return K.inter_solve_torch(i["q"], i["k"], i["g"], i["beta"], i["Akkd"],
                               i["scale"], chunk_size=BT, sub_chunk_size=BC)


def _run_triton(i):
    return K.inter_solve_triton(i["q"], i["k"], i["g"], i["beta"], i["Akkd"],
                                i["scale"], chunk_size=BT, sub_chunk_size=BC)


def _maxdiff(a, b):
    return (a.float() - b.float()).abs().max().item()


# 边界 shape（来自上游小 case 套件，覆盖尾 chunk 不满/奇数 head/K128/多 chunk）
SELFTEST = [
    ("tail_t63",  1,  63, 2, 64),
    ("oddH_k64",  1, 128, 3, 64),
    ("k128_tail", 1, 127, 2, 128),
    ("multi_k128", 2, 100, 3, 128),
    ("long_k64",  1, 1024, 2, 64),
]


def _check_shape(B, T, H, Kk, tag, max_diff):
    i = _build(B, T, H, Kk, "npu:0", seed=20260825)
    tor = _run_torch(i)
    tri = _run_triton(i)
    d = [_maxdiff(a, b) for a, b in zip(tor, tri)]
    ok = all(x < max_diff for x in d)
    print(f"  [{tag:>14}] B{B} T{T:<5} H{H:<3} K{Kk:<3} shape={[B,T,H,Kk]} "
          f"max_diff={max(d):.3e}  {'OK' if ok else 'FAIL'}")
    return ok, max(d)


def _selftest(max_diff):
    allok = True
    for tag, B, T, H, Kk in SELFTEST:
        ok, _ = _check_shape(B, T, H, Kk, tag, max_diff)
        allok &= ok
    return allok


def _target_check(max_diff):
    ok, d = _check_shape(TARGET["B"], TARGET["T"], TARGET["H"], TARGET["K"],
                         "D_KV128_H96_T16384", max_diff)
    return ok


def _perf(repeats, warmup):
    i = _build(TARGET["B"], TARGET["T"], TARGET["H"], TARGET["K"],
               "npu:0", seed=20260825)
    fn = lambda: _run_triton(i)  # noqa: E731
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.npu.synchronize()
    dt = (time.perf_counter() - t0) / repeats
    print(f"[{OP}] triton wall-clock 均值 = {dt*1e3:.3f} ms/调用 "
          f"(warmup={warmup}, repeats={repeats})")
    print("  （wall-clock 非权威；权威 = msprof Task Duration，见 README。）")
    print("  msprof 包裹: msprof --output=./prof_k3 --application=\""
          "python3 test.py --perf --repeats %d --warmup %d\"" % (repeats, warmup))
    return 0


def _report(dirpath):
    hits = []
    for root, _, files in os.walk(dirpath):
        for fn in files:
            if fn.startswith("op_summary_") and fn.endswith(".csv"):
                hits.append(os.path.join(root, fn))
    if not hits:
        print(f"[!] {dirpath!r} 下没有 op_summary_*.csv")
        return 1
    csv_path = max(hits, key=os.path.getmtime)
    with open(csv_path, encoding="utf-8") as f:
        header = next(csv.reader(f))
    i_name = header.index("Op Name")
    i_dur = header.index("Task Duration(us)")
    rows = []
    for line in csv.reader(open(csv_path, encoding="utf-8")):
        if not line or i_name >= len(line):
            continue
        name = line[i_name].strip()
        if not name or "Duration" in name:
            continue
        try:
            dur = float(line[i_dur].replace(",", "").strip())
        except ValueError:
            dur = 0.0
        if dur > 0:
            rows.append((name, dur))
    hit = [(n, d) for n, d in rows if TRITON_NAME in n]
    print(f"# op_summary: {os.path.basename(csv_path)}  (过滤 {TRITON_NAME})")
    if not hit:
        print("[!] 未匹配到 triton kernel 行；实际 op name:")
        for n in sorted({n for n, _ in rows}):
            print("   ", n)
        return 1
    n, tot = len(hit), sum(d for _, d in hit)
    mean_us = tot / n
    print(f"  {hit[0][0]}")
    print(f"  calls={n}  sum={tot:.3f}us  mean={mean_us:.3f} us/调用 = "
          f"{mean_us/1e3:.3f} ms/调用")
    print(f"\n  目标 case 官方隔离基线 ≈14.4–14.9ms/调用（±10%）。"
          f"优化目标是下降这个均值且精度仍 <1e-2。")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="K3 inter_solve 测试/复现脚本")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--target", action="store_true")
    p.add_argument("--perf", action="store_true")
    p.add_argument("--report", metavar="DIR")
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--max-diff", type=float, default=MAX_DIFF)
    a = p.parse_args(argv)

    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        print("[!] 未检测到可用 NPU（在 triton-ascend-env-zhm 容器内先 source ../env.sh）")
        return 2

    print(f"[{OP}] inter_solve 目标 case B{TARGET['B']} T{TARGET['T']} H{TARGET['H']} "
          f"K=V{TARGET['K']}（评分正确性门槛）\n")

    if a.report:
        return _report(a.report)
    if a.perf:
        return _perf(a.repeats, a.warmup)
    if a.selftest:
        ok = _selftest(a.max_diff)
        print(f"\nsummary: selftest={'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    if a.target or True:  # 默认 / --target：只测目标 case（评分门槛）
        ok = _target_check(a.max_diff)
        print("\n=>", "PASS" if ok else "FAIL")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
