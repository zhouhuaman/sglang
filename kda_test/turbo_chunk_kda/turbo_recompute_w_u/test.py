#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""K4 turbo_recompute_w_u 单算子测试脚本（本目录自包含）。

你只需读懂/改动的文件是本目录的 `turbo_recompute_w_u_kernel.py`（含 torch 参考 +
triton kernel）。本脚本负责：造输入 → 正确性(triton vs 同文件 torch 参考) +
性能复现。

用法（在容器内，先 `source ../env.sh`）:
    # ① 目标 case 正确性门槛（评分项，输出 PASS/FAIL 与 max_diff）
    python3 test.py
    # ② 目标 case 性能复现 —— 一条 msprof 指令（测的就是目标 case）+ 解析：
    msprof --output=./prof_k4 --application="python3 test.py --perf --repeats 7 --warmup 3" \\
        && python3 test.py --report ./prof_k4
    #    --perf 只循环 triton（供 msprof 采集）；均值以 op_summary Task Duration(us) 为准。
    # 选看: python3 test.py --selftest   # 边界多 shape 快速自检（迭代期用，非评分项）

正确性判定: 三个输出 (w, u, kg) triton-vs-torch max_diff < 1e-2 即 OK
（K4 目标 case 约 1e-6 量级）。
性能口径: msprof `Task Duration(us)` 每调用均值。目标 case 官方基线与门槛见 ../TEST_REPORT.md
（±10%）；优化目标是让这个数字下降且精度不劣于 1e-2。
"""

import argparse
import csv
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import torch  # noqa: E402
import torch_npu  # noqa: F401
import turbo_recompute_w_u_kernel as K  # noqa: E402

TARGET = dict(B=1, T=16384, H=96, K=128, V=128)
MAX_DIFF = 1e-2
TRITON_NAME = "_recompute_w_u_kernel"
BT = 64
OP = "k4"


def _build(B, T, H, Kk, V, device, seed):
    torch.manual_seed(seed)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, Kk), dim=-1).to(device)
    v = torch.nn.functional.normalize(torch.randn(B, T, H, V), dim=-1).to(device)
    beta = torch.rand(B, T, H).sigmoid().to(device)
    gk = (torch.randn(B, T, H, Kk) * 0.5 - 2.0).to(device)  # log2 空间 gate cumsum
    # A（Akk_inv）: 逐 chunk 的单位下三角矩阵（真实形态）。每 chunk 一个随机
    # [BT, BT] 单位下三角，行主序铺到 A[b, chunk*BT+i, h, :] = M[b,chunk,h][i, :]。
    NT = -(T // -BT)
    M = torch.zeros(B, NT, BT, H, BT, device=device)
    M += torch.tril(torch.randn(B, NT, BT, H, BT, device=device) * 0.2)
    M = torch.tril(M, diagonal=0)
    M = M + torch.eye(BT, device=device).view(1, 1, BT, 1, BT)  # 单位下三角（含对角 1）
    M = M.masked_fill(torch.triu(torch.ones(BT, BT, device=device), diagonal=1).bool()
                      .view(1, 1, BT, 1, BT), 0.0)
    A = M.reshape(B, NT * BT, H, BT)[:, :T].contiguous()
    return dict(k=k, v=v, beta=beta, A=A, gk=gk)


def _run_torch(i):
    return K.turbo_recompute_w_u_torch(i["k"], i["v"], i["beta"], i["A"], i["gk"],
                                 chunk_size=BT)


def _run_triton(i):
    return K.turbo_recompute_w_u_triton(i["k"], i["v"], i["beta"], i["A"], i["gk"],
                                  chunk_size=BT)


def _maxdiff(a, b):
    return (a.float() - b.float()).abs().max().item()


# 边界 shape：覆盖尾 chunk 不满 / K=V=32 / 128 / 多 head / 多 chunk / 多 batch
SELFTEST = [
    ("tiny_k64", 1,  64, 1, 64, 64),
    ("tail_t63", 1,  63, 2, 64, 64),
    ("k128_tail",1, 127, 2, 128, 128),
    ("k32_long", 1, 193, 1, 32, 32),
    ("h8_k64",   1, 256, 8, 64, 64),
    ("kv_mix",   1, 128, 2, 64, 128),
    ("b2_k64",   2, 128, 2, 64, 64),
]


def _check_shape(B, T, H, Kk, V, tag, max_diff):
    i = _build(B, T, H, Kk, V, "npu:0", seed=20260825)
    tor = _run_torch(i)
    tri = _run_triton(i)
    d = [_maxdiff(a, b) for a, b in zip(tor, tri)]
    ok = all(x < max_diff for x in d)
    print(f"  [{tag:>14}] B{B} T{T:<5} H{H:<3} K{Kk:<3} V{V:<3} shape={[B,T,H,Kk,V]} "
          f"max_diff={max(d):.3e}  {'OK' if ok else 'FAIL'}")
    return ok, max(d)


def _selftest(max_diff):
    allok = True
    for tag, B, T, H, Kk, V in SELFTEST:
        ok, _ = _check_shape(B, T, H, Kk, V, tag, max_diff)
        allok &= ok
    return allok


def _target_check(max_diff):
    ok, d = _check_shape(TARGET["B"], TARGET["T"], TARGET["H"], TARGET["K"],
                         TARGET["V"], "D_KV128_H96_T16384", max_diff)
    return ok


def _perf(repeats, warmup):
    i = _build(TARGET["B"], TARGET["T"], TARGET["H"], TARGET["K"], TARGET["V"],
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
    print("  msprof 包裹: msprof --output=./prof_k4 --application=\""
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
    print("\n  目标 case 官方基线与门槛见 ../TEST_REPORT.md。"
          "优化目标是下降这个均值且精度仍 <1e-2。")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="K4 turbo_recompute_w_u 测试/复现脚本")
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

    print(f"[{OP}] recompute_w_u 目标 case B{TARGET['B']} T{TARGET['T']} H{TARGET['H']} "
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
