#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""K6 gla_output interview-question test script (self-contained in this directory).

The only file you need to read/modify is this directory's `gla_output_kernel.py`
(torch reference + triton kernel). This script is responsible for: building inputs ->
correctness (triton vs the same file's torch reference) + performance reproduction.

Usage (in the container, after `source ../env.sh`):
    # 1. target-case correctness gate (scored: prints PASS/FAIL and max_diff)
    python3 test.py
    # 2. target-case performance reproduction — one msprof command (profiles exactly the
    #    target case) + parsing:
    msprof --output=./prof_k6 --application="python3 test.py --perf --repeats 7 --warmup 3" \\
        && python3 test.py --report ./prof_k6
    #    --perf loops only the triton kernel (for msprof capture); the mean comes from the
    #    op_summary Task Duration(us) column.
    # optional: python3 test.py --selftest   # quick multi-shape self-check of boundary
    # shapes (for the iteration phase; not scored)

Correctness gate: output o triton-vs-torch max_diff < 1e-2 is OK (the K6 target case is
~1e-6 magnitude). Performance basis: per-call mean of msprof `Task Duration(us)`. The
target case's official isolated baseline is in PROBLEM.md (±10%); the optimization goal
is to lower that number while keeping accuracy no worse than 1e-2.
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
import gla_output_kernel as K  # noqa: E402

TARGET = dict(B=1, T=16384, H=96, K=128, V=128)
MAX_DIFF = 1e-2
# Target case H%16==0 -> the head-merged `chunk_gla_fwd_kernel_o_hm`; small H falls back
# to `chunk_gla_fwd_kernel_o`. Both share a prefix, so filter by prefix (substring match).
TRITON_NAME = "chunk_gla_fwd_kernel_o"
BT = 64
OP = "k6"


def _build(B, T, H, Kk, V, device, seed):
    torch.manual_seed(seed)
    NT = -(T // -BT)
    q = torch.nn.functional.normalize(torch.randn(B, T, H, Kk), dim=-1).to(device)
    v_new = torch.nn.functional.normalize(torch.randn(B, T, H, V), dim=-1).to(device)
    g = (torch.randn(B, T, H, Kk) * 0.5 - 2.0).to(device)   # cumulative gate in log2 space (negative)
    # Aqk: intra-chunk causal attention weights (row t, col = position within the chunk).
    # Only the lower triangle is meaningful in real values (both the torch reference and
    # the triton kernel drop the upper triangle via a tril mask), so give fully random
    # values here to cover the masking path.
    Aqk = (torch.randn(B, T, H, BT) * 0.1).to(device)
    # h: compressed-state snapshot at the start of each chunk [B, NT, H, V, K]
    h = (torch.randn(B, NT, H, V, Kk) * 0.3).to(device)
    scale = 1.0 / (Kk ** 0.5)
    return dict(q=q, v_new=v_new, g=g, Aqk=Aqk, h=h, scale=scale)


def _run_torch(i):
    return [K.gla_output_torch(i["q"], i["v_new"], i["g"], i["Aqk"], i["h"],
                               i["scale"], chunk_size=BT)]


def _run_triton(i):
    return [K.gla_output_triton(i["q"], i["v_new"], i["g"], i["Aqk"], i["h"],
                                i["scale"], chunk_size=BT)]


def _maxdiff(a, b):
    return (a.float() - b.float()).abs().max().item()


# Boundary shapes: trailing chunk shorter than BT / K=V=32 / 128 / V!=K / multiple heads
# (H=8) / multiple chunks / multiple batches
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
    print(f"[{OP}] triton wall-clock mean = {dt*1e3:.3f} ms/call "
          f"(warmup={warmup}, repeats={repeats})")
    print("  (wall-clock is not authoritative; authoritative = msprof Task Duration, see README.)")
    print("  msprof wrapper: msprof --output=./prof_k6 --application=\""
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
    print(f"# op_summary: {os.path.basename(csv_path)}  (filtered by {TRITON_NAME})")
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
    print("\n  The target case's official isolated baseline is in PROBLEM.md; "
          "the optimization goal is to lower this mean while keeping accuracy <1e-2.")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="K6 gla_output test/reproduction script")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--target", action="store_true")
    p.add_argument("--perf", action="store_true")
    p.add_argument("--report", metavar="DIR")
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--max-diff", type=float, default=MAX_DIFF)
    a = p.parse_args(argv)

    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        print("[!] no usable NPU detected (in the triton-ascend-env-zhm container, source ../env.sh first)")
        return 2

    print(f"[{OP}] gla_output target case B{TARGET['B']} T{TARGET['T']} H{TARGET['H']} "
          f"K=V{TARGET['K']} (scored correctness gate)\n")

    if a.report:
        return _report(a.report)
    if a.perf:
        return _perf(a.repeats, a.warmup)
    if a.selftest:
        ok = _selftest(a.max_diff)
        print(f"\nsummary: selftest={'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    if a.target or True:  # default / --target: test only the target case (scored gate)
        ok = _target_check(a.max_diff)
        print("\n=>", "PASS" if ok else "FAIL")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
