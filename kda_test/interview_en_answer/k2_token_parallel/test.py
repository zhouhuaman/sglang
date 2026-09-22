#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""K2 token_parallel interview test script (this directory is self-contained).

The only file you need to read/modify is `token_parallel_kernel.py` in this directory
(torch reference + triton kernel). This script handles: build inputs → correctness
(triton vs the same file's torch reference) + performance reproduction.

Usage (inside the container, first `source ../env.sh`):
    # ① target-case correctness gate (scored; prints PASS/FAIL and max_diff)
    python3 test.py
    # ② target-case performance reproduction -- a single msprof command (times exactly the
    #    target case) + parsing:
    msprof --output=./prof_k2 --application="python3 test.py --perf --repeats 7 --warmup 3" \\
        && python3 test.py --report ./prof_k2
    #    --perf loops only the triton kernel (for msprof capture); the mean is taken from
    #    the op_summary Task Duration(us).
    # Optional: python3 test.py --selftest   # quick multi-shape edge self-check (dev-time only, not scored)

Correctness criterion: triton-vs-torch max_diff < 1e-2 on each output is OK
(K2 target-case baseline is ~1e-7).
Performance metric: per-call mean of the msprof `Task Duration(us)`. The official isolated
baseline for the target case is ~9.5–9.8ms/call (±10%); the optimization goal is to bring
this number down while keeping precision no worse than 1e-2.
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
import token_parallel_kernel as K  # noqa: E402

TARGET = dict(B=1, T=16384, H=96, K=128, V=128)
MAX_DIFF = 1e-2
TRITON_NAME = "_token_parallel_kernel"
BT = 64
BC = 16
OP = "k2"


def _base_inputs(B, T, H, Kk, device, seed):
    torch.manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(B, T, H, Kk), dim=-1).to(device)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, Kk), dim=-1).to(device)
    g = (torch.randn(B, T, H, Kk) * 0.5 - 2.0).to(device)  # log2-space gate
    beta = torch.rand(B, T, H).sigmoid().to(device)
    return q, k, g, beta


def _build(B, T, H, Kk, device, seed):
    q, k, g, beta = _base_inputs(B, T, H, Kk, device, seed)
    return dict(q=q, k=k, gk=g, beta=beta, scale=1.0 / (Kk ** 0.5))


def _run_torch(i):
    return K.token_parallel_torch(i["q"], i["k"], i["gk"], i["beta"], i["scale"],
                                  chunk_size=BT, sub_chunk_size=BC)


def _run_triton(i):
    return K.token_parallel_triton(i["q"], i["k"], i["gk"], i["beta"], i["scale"],
                                   chunk_size=BT, sub_chunk_size=BC)


def _maxdiff(a, b):
    return (a.float() - b.float()).abs().max().item()


# Edge shapes: cover a trailing/partial chunk / K=128 / K=32 / multiple heads (H=8) / multiple chunks
SELFTEST = [
    ("tiny_k64", 1, 128, 2, 64),
    ("tail_t63", 1,  63, 2, 64),
    ("k128_tail", 1, 127, 2, 128),
    ("k32",      1, 193, 1, 32),
    ("h8_k64",   1, 256, 8, 64),
    ("long_k64", 1, 2562, 1, 64),
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
    print(f"[{OP}] triton wall-clock mean = {dt*1e3:.3f} ms/call "
          f"(warmup={warmup}, repeats={repeats})")
    print("  (wall-clock is not authoritative; the authoritative one is the msprof Task Duration, see README.)")
    print("  msprof wrapper: msprof --output=./prof_k2 --application=\""
          "python3 test.py --perf --repeats %d --warmup %d\"" % (repeats, warmup))
    return 0


def _report(dirpath):
    hits = []
    for root, _, files in os.walk(dirpath):
        for fn in files:
            if fn.startswith("op_summary_") and fn.endswith(".csv"):
                hits.append(os.path.join(root, fn))
    if not hits:
        print(f"[!] no op_summary_*.csv under {dirpath!r}")
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
    print(f"# op_summary: {os.path.basename(csv_path)}  (filtering {TRITON_NAME})")
    if not hit:
        print("[!] no triton kernel row matched; actual op names:")
        for n in sorted({n for n, _ in rows}):
            print("   ", n)
        return 1
    n, tot = len(hit), sum(d for _, d in hit)
    mean_us = tot / n
    print(f"  {hit[0][0]}")
    print(f"  calls={n}  sum={tot:.3f}us  mean={mean_us:.3f} us/call = "
          f"{mean_us/1e3:.3f} ms/call")
    print("\n  Target case: official isolated baseline ≈9.5–9.8ms/call (±10%). "
          "Optimization goal: lower this mean while keeping precision <1e-2.")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="K2 token_parallel test/reproduction script")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--target", action="store_true")
    p.add_argument("--perf", action="store_true")
    p.add_argument("--report", metavar="DIR")
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--max-diff", type=float, default=MAX_DIFF)
    a = p.parse_args(argv)

    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        print("[!] no usable NPU detected (source ../env.sh first inside the triton-ascend-env-zhm container)")
        return 2

    print(f"[{OP}] token_parallel target case B{TARGET['B']} T{TARGET['T']} H{TARGET['H']} "
          f"K=V{TARGET['K']} (scored correctness gate)\n")

    if a.report:
        return _report(a.report)
    if a.perf:
        return _perf(a.repeats, a.warmup)
    if a.selftest:
        ok = _selftest(a.max_diff)
        print(f"\nsummary: selftest={'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    if a.target or True:  # Default / --target: test only the target case (scored gate)
        ok = _target_check(a.max_diff)
        print("\n=>", "PASS" if ok else "FAIL")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
