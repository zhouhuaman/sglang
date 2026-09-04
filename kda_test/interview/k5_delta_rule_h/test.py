#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""K5 delta_rule_h 面试题测试脚本（本目录自包含）。

你只需读懂/改动的文件是本目录的 `delta_rule_h_kernel.py`（含 torch 参考 +
triton kernel）。本脚本负责：造输入 → 正确性(triton vs 同文件 torch 参考) +
性能复现。

用法（在容器内，先 `source ../env.sh`）:
    # ① 目标 case 正确性门槛（评分项，输出 PASS/FAIL 与 max_diff；torch 参考较慢，
    #    96 head × 256 chunk 串行 matmul，约 10–20s，属正常，请耐心）
    python3 test.py
    # ② 目标 case 性能复现 —— 一条 msprof 指令（测的就是目标 case）+ 解析：
    msprof --output=./prof_k5 --application="python3 test.py --perf --repeats 7 --warmup 3" \\
        && python3 test.py --report ./prof_k5
    #    --perf 只循环 triton（供 msprof 采集）；均值以 op_summary Task Duration(us) 为准。
    # 选看: python3 test.py --selftest   # 边界多 shape 快速自检（迭代期用，非评分项）

正确性判定: 每个输出 triton-vs-torch max_diff < 1e-2 即 OK（目标 case 基线约 1e-7）。
性能口径: msprof `Task Duration(us)` 每调用均值。目标 case 官方隔离基线约
9.1–9.5ms/调用（±10%）；优化目标是让这个数字下降且精度不劣于 1e-2。
* torch 与 triton 两版都会 **in-place 更新 initial_state**，脚本每次调用都传
  clone，避免相互污染；也请你改 kernel 时保持「每次跑前 clone」的习惯。
* K==V 约束（上游 kernel 同要求）；本目录只测 K==V 用例。
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
import delta_rule_h_kernel as K  # noqa: E402

TARGET = dict(B=1, T=16384, H=96, K=128, V=128)
MAX_DIFF = 1e-2
TRITON_NAME = "_delta_rule_h_kernel"
BT = 64
OP = "k5"


def _build(B, T, H, Kk, device, seed):
    """合成输入，分布与上游 test_level2_kernel_precision 一致（K==V）。

    量级贴合真实链路: k L2-归一化; w/u 是 Kernel-4 输出 ~0.1; gk 在 log2 空间
    (chunk-local cumsum + RCP_LN2 缩放后典型 -2.6~+1.3); initial_state ~0.05。
    kernel 运行时间只依赖 shape/mask、不依赖取值，故该分布可复现官方基线。
    """
    torch.manual_seed(seed)
    V = Kk
    q = torch.nn.functional.normalize(torch.randn(B, T, H, Kk), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, Kk), dim=-1)
    w = torch.randn(B, T, H, Kk) * 0.1
    u = torch.randn(B, T, H, V) * 0.1
    gk = torch.randn(B, T, H, Kk) * 0.5 - 2.0
    initial_state = torch.randn(B, H, V, Kk) * 0.05
    indices = torch.arange(B, dtype=torch.int32)
    return dict(k=k, w=w, u=u, gk=gk, initial_state=initial_state, indices=indices)


def _to_npu(d, device):
    return {n: t.to(device) for n, t in d.items()}


def _run_torch(i):
    # 每次 clone initial_state（两版都 in-place 更新）
    return K.delta_rule_h_torch(
        i["k"], i["w"], i["u"], i["gk"], i["initial_state"].clone(),
        i["indices"], chunk_size=BT)


def _run_triton(i):
    return K.delta_rule_h_triton(
        i["k"], i["w"], i["u"], i["gk"], i["initial_state"].clone(),
        i["indices"], chunk_size=BT)


def _maxdiff(a, b):
    return (a.float() - b.float()).abs().max().item()


# 边界 shape：覆盖尾 chunk 不满 / 奇数 head / 多 batch / K=64 flat-store 与
# K=128 2D-store 两条路径 / 长序列多 chunk 串行递推
SELFTEST = [
    ("tiny_default", 1, 128, 2, 64),
    ("tail_t63",     1,  63, 2, 64),
    ("oddH_t128",    1, 128, 3, 64),
    ("multi_B2H3",   2, 100, 3, 64),
    ("k128_path",    1, 128, 2, 128),
    ("long_t2562",   1, 2562, 1, 64),
    ("tail_t255",    1, 255, 2, 64),
]


def _check_shape(B, T, H, Kk, tag, max_diff):
    i = _to_npu(_build(B, T, H, Kk, "cpu", seed=20260825), "npu:0")
    a = dict(i)
    a["initial_state"] = i["initial_state"].clone()
    b = dict(i)
    b["initial_state"] = i["initial_state"].clone()
    h_t, v_t = _run_torch(a)
    h_r, v_r = _run_triton(b)
    # 对比 (h, v_new) 以及两版各自 in-place 写回的最终 initial_state
    d = [
        _maxdiff(h_t, h_r),
        _maxdiff(v_t, v_r),
        _maxdiff(a["initial_state"], b["initial_state"]),
    ]
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
    base = _to_npu(_build(TARGET["B"], TARGET["T"], TARGET["H"], TARGET["K"],
                          "cpu", seed=20260825), "npu:0")
    fn = lambda: _run_triton(base)  # noqa: E731  (内部 clone，不累积状态)
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
    print("  msprof 包裹: msprof --output=./prof_k5 --application=\""
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
    print("\n  目标 case 官方隔离基线 ≈9.1–9.5ms/调用（±10%）。"
          "优化目标是下降这个均值且精度仍 <1e-2。")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="K5 delta_rule_h 测试/复现脚本")
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

    print(f"[{OP}] delta_rule_h 目标 case B{TARGET['B']} T{TARGET['T']} "
          f"H{TARGET['H']} K=V{TARGET['K']}（评分正确性门槛）\n")

    if a.report:
        return _report(a.report)
    if a.perf:
        return _perf(a.repeats, a.warmup)
    if a.selftest:
        ok = _selftest(a.max_diff)
        print(f"\nsummary: selftest={'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    if a.target or True:  # 默认 / --target：只测目标 case（评分门槛；torch 参考约 10–20s）
        ok = _target_check(a.max_diff)
        print("\n=>", "PASS" if ok else "FAIL")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
