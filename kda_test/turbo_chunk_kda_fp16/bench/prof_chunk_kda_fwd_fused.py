#!/usr/bin/env python3
"""Standalone perf test for vllm-ascend fused ChunkKdaFwd (ASCENDC op).

Target case (D_KV128_H96_T16384): B=1, T=16384, H=96, K=128, V=128, layout=BSND.

Two timing sources:
  * event timing (torch.npu Event) -- quick reference;
  * msprof / msopprof (op_summary CSV parse) -- authoritative, per design.md:
    "性能结论只使用 msopprof".

Usage:
  # event timing
  python3 prof_chunk_kda_fwd_fused.py --dtype fp16 --chunk-size 64 --iters 100

  # msprof capture (run whole script under msprof, then parse)
  msprof --application="python3 prof_chunk_kda_fwd_fused.py --iters 200" \
         --output=/tmp/kda_prof
  python3 prof_chunk_kda_fwd_fused.py --parse /tmp/kda_prof

  # optional quick correctness spot check at reduced T (same H/K/V/chunk path)
  python3 prof_chunk_kda_fwd_fused.py --check
"""
import argparse
import csv
import glob
import os
import sys

# This machine has two editable installs of vllm_ascend; the stale one
# (other user's checkout) wins via site-packages editable finders.
# Pin to THIS repo explicitly (regular sys.path beats editable finders).
_REPO = "/data/autotriton/sekd/vllm-ascend-community/vllm-ascend"
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch
import torch_npu

# ---- load vllm-ascend custom op extension directly (bypasses A5 runtime gate) ----
from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env(include_vendor_lib=True)
import vllm_ascend.vllm_ascend_C  # noqa: E402,F401  (registers torch.ops._C_ascend.*)

torch_npu.npu.config.allow_internal_format = True

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
OP_NAME = "ChunkKdaFwd"


def make_inputs(b, t, h, k, v, dtype, seed=42):
    torch.manual_seed(seed)
    q = (torch.randn(b, t, h, k, dtype=dtype) * 0.05).npu()
    kk = (torch.randn(b, t, h, k, dtype=dtype) * 0.05).npu()
    vv = (torch.randn(b, t, h, v, dtype=dtype) * 0.05).npu()
    g = (-torch.rand(b, t, h, k, dtype=torch.float32) * 0.05).npu()
    beta = torch.sigmoid(torch.randn(b, t, h, dtype=torch.float32)).npu()
    return q, kk, vv, g, beta


def run_op(q, k, v, g, beta, scale, chunk_size, **kw):
    return torch.ops._C_ascend.chunk_kda_fwd(
        q, k, v, g, beta, scale, chunk_size, layout="BSND", **kw
    )


def event_timing(q, k, v, g, beta, scale, chunk_size, warmup, iters, **kw):
    run_op(q, k, v, g, beta, scale, chunk_size, **kw)
    torch.npu.synchronize()
    for _ in range(warmup):
        run_op(q, k, v, g, beta, scale, chunk_size, **kw)
    torch.npu.synchronize()

    times = []
    s, e = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    for _ in range(iters):
        s.record()
        run_op(q, k, v, g, beta, scale, chunk_size, **kw)
        e.record()
        torch.npu.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    n = len(times)
    return {
        "avg_ms": sum(times) / n,
        "min_ms": times[0],
        "median_ms": times[n // 2],
        "max_ms": times[-1],
    }


def parse_op_summary(out_dir):
    """Parse msprof op_summary_*.csv.

    On A5 the aclnn call launches 1 layout Transpose + 4 ChunkKdaFwd kernels
    (kernels named ``aclnnChunkKdaFwd_*``). Group by Transpose boundary and
    report per-call totals as well as per-kernel-name breakdown.
    """
    hits = []
    for root, _, files in os.walk(out_dir):
        for fn in files:
            if fn.startswith("op_summary_") and fn.endswith(".csv"):
                hits.append(os.path.join(root, fn))
    if not hits:
        raise SystemExit(f"[!] no op_summary_*.csv under {out_dir!r}")
    rows = []
    seen_names = {}
    for p in sorted(hits):
        with open(p, encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
        i_name = header.index("Op Name")
        i_dur = header.index("Task Duration(us)")
        with open(p, encoding="utf-8") as f:
            for line in csv.reader(f):
                if not line or len(line) <= max(i_name, i_dur):
                    continue
                name = line[i_name].strip()
                if not name or name == "Op Name":
                    continue
                seen_names[name] = seen_names.get(name, 0) + 1
                if "ChunkKdaFwd" in name:
                    rows.append((name, float(line[i_dur])))
    if not rows:
        raise SystemExit(f"[!] op '{OP_NAME}' not found in op_summary; seen names={seen_names}")

    # per-kernel-name stats
    per_name = {}
    for name, d in rows:
        per_name.setdefault(name, []).append(d)
    per_name_stats = {}
    for name, ds in per_name.items():
        ds.sort()
        n = len(ds)
        per_name_stats[name] = {
            "n": n, "avg_us": sum(ds) / n, "min_us": ds[0],
            "median_us": ds[n // 2], "max_us": ds[-1],
        }

    # per-call totals: a call = 1 Transpose row followed by its kernels
    calls, cur = [], []
    for name, d in rows:
        if name.startswith("aclnnChunkKdaFwd_Transpose"):
            if cur:
                calls.append(cur)
                cur = []
        else:
            cur.append(d)
    if cur:
        calls.append(cur)
    totals = sorted(sum(c) for c in calls)
    n = len(totals)
    return {
        "n_calls": n,
        "avg_us": sum(totals) / n,
        "min_us": totals[0],
        "median_us": totals[n // 2],
        "max_us": totals[-1],
        "kernels_per_call": sorted(set(len(c) for c in calls)),
        "per_name": per_name_stats,
    }


# ---------------- correctness spot check (reduced T, same shape path) ----------------
def _gate_cumsum_reference(g, chunk_size):
    g_cpu = g.detach().cpu().to(torch.float32)
    ref = torch.empty_like(g_cpu)
    rcp_ln2 = 1.4426950408889634
    for start in range(0, g_cpu.shape[1], chunk_size):
        end = min(start + chunk_size, g_cpu.shape[1])
        ref[:, start:end] = torch.cumsum(g_cpu[:, start:end] * rcp_ln2, dim=1)
    return ref


def _lower_inverse(mat):
    eye = torch.eye(mat.shape[-1], device=mat.device, dtype=torch.float32)
    lhs = eye + torch.tril(mat.to(torch.float32), diagonal=-1)
    return torch.linalg.solve_triangular(lhs, eye, upper=False)


def chunk_kda_forward_reference(q, k, v, gk, beta, scale, chunk_size):
    bsz, total_t, hq, kdim = q.shape
    _, _, hv, vdim = v.shape
    group = hv // hq
    out_dtype = v.dtype
    device = q.device
    o = torch.zeros((bsz, total_t, hv, vdim), device=device, dtype=out_dtype)
    state = torch.zeros((bsz, hv, kdim, vdim), device=device, dtype=torch.float32)
    for b in range(bsz):
        for start in range(0, total_t, chunk_size):
            end = min(start + chunk_size, total_t)
            cur_t = end - start
            for ihv in range(hv):
                ih = ihv // group
                q_blk = q[b, start:end, ih].to(torch.float32)
                k_blk = k[b, start:end, ih].to(torch.float32)
                v_blk = v[b, start:end, ihv].to(torch.float32)
                g_blk = gk[b, start:end, ihv].to(torch.float32)
                beta_blk = beta[b, start:end, ihv].to(torch.float32)

                causal = torch.ones((cur_t, cur_t), device=device, dtype=torch.bool).tril()
                strict_causal = torch.ones((cur_t, cur_t), device=device, dtype=torch.bool).tril(diagonal=-1)
                rel = g_blk[:, None, :] - g_blk[None, :, :]
                rel = rel.masked_fill(~causal[:, :, None], 0.0)
                gate = torch.exp2(rel)
                qk = torch.einsum("ik,jk,ijk->ij", q_blk, k_blk, gate) * float(scale)
                kk = torch.einsum("ik,jk,ijk->ij", k_blk, k_blk, gate)
                tril_qk = torch.where(causal, qk, torch.zeros_like(qk))
                tril_kk = torch.where(strict_causal, kk * beta_blk[:, None], torch.zeros_like(kk))
                inv_akk = _lower_inverse(tril_kk)

                k_beta_g = k_blk * beta_blk[:, None] * torch.exp2(g_blk)
                v_beta = v_blk * beta_blk[:, None]
                w_blk = inv_akk @ k_beta_g
                u_blk = inv_akk @ v_beta

                last_g = g_blk[cur_t - 1]
                qg_blk = q_blk * torch.exp2(g_blk)
                kg_blk = k_blk * torch.exp2(last_g[None, :] - g_blk)
                h_prev = state[b, ihv].clone()
                v_new_blk = u_blk - w_blk @ h_prev
                state[b, ihv] = torch.exp2(last_g)[:, None] * h_prev + kg_blk.T @ v_new_blk

                o_inter = qg_blk @ h_prev * float(scale)
                o_local = tril_qk @ v_new_blk
                o[b, start:end, ihv] = (o_inter + o_local).to(out_dtype)
    return o


def spot_check(h, k, v, chunk_size, dtype):
    t = 1024  # reduced T: same H/K/V/chunk path, CPU reference stays fast
    q, kk, vv, g, beta = make_inputs(1, t, h, k, v, dtype, seed=7)
    scale = k ** -0.5
    got = run_op(q, kk, vv, g, beta, scale, chunk_size)[0].float()
    gk = _gate_cumsum_reference(g, chunk_size)
    ref = chunk_kda_forward_reference(
        q.cpu(), kk.cpu(), vv.cpu(), gk, beta.cpu(), scale=scale, chunk_size=chunk_size
    )
    diff = (got.cpu() - ref).abs()
    rel = diff / (ref.abs() + 1e-6)
    print(f"[check] T={t} H={h} K={k} V={v} chunk={chunk_size} {dtype}: "
          f"max_abs={diff.max().item():.3e} max_rel={rel.max().item():.3e}")
    assert diff.max().item() < 2e-2 * max(1.0, ref.abs().max().item()), "spot check FAILED"
    print("[check] OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=1)
    ap.add_argument("--t", type=int, default=16384)
    ap.add_argument("--h", type=int, default=96)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--v", type=int, default=128)
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="fp16")
    ap.add_argument("--chunk-size", type=int, default=64, choices=(64, 128))
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--use-gate-in-kernel", action="store_true")
    ap.add_argument("--check", action="store_true", help="run correctness spot check at reduced T")
    ap.add_argument("--parse", metavar="DIR", help="parse msprof output dir instead of timing")
    args = ap.parse_args()

    if args.parse:
        r = parse_op_summary(args.parse)
        print(f"[msprof] {r['n_calls']} calls, {r['kernels_per_call']} kernels per call: "
              f"per-call avg={r['avg_us']:.1f}us min={r['min_us']:.1f}us "
              f"median={r['median_us']:.1f}us max={r['max_us']:.1f}us")
        for name, s in r["per_name"].items():
            print(f"[msprof]   {name}: n={s['n']} avg={s['avg_us']:.1f}us "
                  f"min={s['min_us']:.1f} median={s['median_us']:.1f} max={s['max_us']:.1f}")
        return

    if args.check:
        spot_check(args.h, args.k, args.v, args.chunk_size, DTYPES[args.dtype])
        return

    dtype = DTYPES[args.dtype]
    q, k, v, g, beta = make_inputs(args.b, args.t, args.h, args.k, args.v, dtype)
    scale = args.k ** -0.5
    kw = {}
    if args.use_gate_in_kernel:
        kw.update(use_gate_in_kernel=True, A_log=torch.zeros(args.h, dtype=torch.float32).npu())

    r = event_timing(q, k, v, g, beta, scale, args.chunk_size, args.warmup, args.iters, **kw)
    tokens = args.b * args.t
    print(f"[event] B={args.b} T={args.t} H={args.h} K={args.k} V={args.v} "
          f"dtype={args.dtype} chunk={args.chunk_size} iters={args.iters}")
    print(f"[event] avg={r['avg_ms'] * 1000:.1f}us min={r['min_ms'] * 1000:.1f}us "
          f"median={r['median_ms'] * 1000:.1f}us max={r['max_ms'] * 1000:.1f}us")
    print(f"[event] throughput: {tokens / r['avg_ms'] / 1000:.1f} tokens/ms "
          f"({tokens / r['avg_ms'] * 1000 / 1e3:.1f} K tokens/s)")
    print(f"[event] run under msprof for authoritative kernel time:\n"
          f"  msprof --application=\"python3 {os.path.abspath(__file__)} --iters 200\" "
          f"--output=/tmp/kda_prof\n"
          f"  python3 {os.path.abspath(__file__)} --parse /tmp/kda_prof")


if __name__ == "__main__":
    main()
