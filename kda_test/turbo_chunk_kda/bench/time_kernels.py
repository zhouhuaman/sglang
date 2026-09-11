#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""KDA 6 算子逐 kernel wall-clock 计时 + 精度（迭代优化用）。

背景: 当前 CANN 的 msprof ``--export=on`` 只产 sqlite、不再产
per_case_profile.py 需要的 op_summary_*.csv（导出回归）。迭代优化期间改用
wall-clock 计时（``time.perf_counter`` + ``torch.npu.synchronize``），口径与
per-op ``run.py::_timeit`` 一致；最终报告数字仍用 msprof（README.md §7）。

对目标 case 的每个 kernel 在**完全相同的输入**（bench.py::_derive_inputs 的
torch 链中间量）下分别测 K_torch / K_triton 的 mean 时长，并复核精度
(max_diff)。输出表 + ``time_results.json`` 供迭代对比。

用法（须在容器 triton-ascend-env-zhm 内，同 run_cpu.sh 的环境）:
    python3 time_kernels.py [--repeats 5] [--warmup 2] [--device npu:0]
    python3 time_kernels.py --kernel K3        # 只测单 kernel（改完快速验证）
    python3 time_kernels.py --json out.json    # 输出路径
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import bench  # noqa: E402


def _mean_time(fn, warmup, repeats):
    """wall-clock mean 耗时（us）。warmup 不计时，repeats 取均值。"""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        times.append((time.perf_counter() - t0) * 1e6)
    return sum(times) / len(times)


def derive_prefix(kid, base, device):
    """只派生到 kid 所需的 torch 链前缀（避免全链 K5 torch ~15s 的开销）。

    与 bench._derive_inputs 的前缀段一致；返回 kernels_in（含到 kid 的条目）。
    调用方只测 kid 时用此代替全链派生。
    """
    x = base["x"].to(device)
    A_log = base["A_log"].to(device)
    dt_bias = base["dt_bias"].to(device)
    q = base["q"].to(device)
    k = base["k"].to(device)
    v = base["v"].to(device)
    beta = base["beta"].to(device)
    scale = base["scale"]
    BT = bench._BT
    BC = bench._BC
    kernels_in = {}

    if kid in ("K2", "K3", "K4", "K5", "K6"):
        g = bench.k1_torch(x, A_log, dt_bias=dt_bias, chunk_size=BT, scale=bench.RCP_LN2)
    if kid in ("K3", "K4", "K5", "K6"):
        Aqk_d, Akk = bench.k2_torch(q, k, g, beta, scale, chunk_size=BT, sub_chunk_size=BC)
    if kid in ("K4", "K5", "K6"):
        Aqk_nd, Akk_inv = bench.k3_torch(q, k, g, beta, Akkd=Akk, scale=scale,
                                         chunk_size=BT, sub_chunk_size=BC)
    if kid in ("K5", "K6"):
        w, u, kg = bench.k4_torch(k, v, beta, A=Akk_inv, gk=g, chunk_size=BT)
    if kid == "K6":
        B, T, H, K_ = k.shape
        V = v.shape[-1]
        initial_state = (torch.randn(B, H, V, K_, dtype=torch.float32) * 0.05).to(device)
        initial_state_indices = torch.arange(B, dtype=torch.int32).to(device)
        init_backup = initial_state.detach().clone()
        h_setup, v_new_setup = bench.k5_torch(
            kg, w, u, gk=g, initial_state=initial_state.detach().clone(),
            initial_state_indices=initial_state_indices, chunk_size=BT)
        Aqk_merge = Aqk_d + Aqk_nd

    # 与 bench._derive_inputs 的组装逐条一致
    if kid in ("K1", "K2", "K3", "K4", "K5", "K6"):
        kernels_in["K1"] = {
            "torch_args": (x, A_log, dt_bias, BT, bench.RCP_LN2), "torch_kwargs": {},
            "triton_args": (x, A_log, dt_bias, BT, bench.RCP_LN2), "triton_kwargs": {},
        }
    if kid in ("K2", "K3", "K4", "K5", "K6"):
        kernels_in["K2"] = {
            "torch_args": (q, k, g, beta, scale, BT, BC), "torch_kwargs": {},
            "triton_args": (q, k, g, beta, scale),
            "triton_kwargs": {"chunk_size": BT, "sub_chunk_size": BC},
        }
    if kid in ("K3", "K4", "K5", "K6"):
        kernels_in["K3"] = {
            "torch_args": (q, k, g, beta, Akk, scale, BT, BC), "torch_kwargs": {},
            "triton_args": (q, k, g, beta, Akk, scale),
            "triton_kwargs": {"chunk_size": BT, "sub_chunk_size": BC},
        }
    if kid in ("K4", "K5", "K6"):
        kernels_in["K4"] = {
            "torch_args": (k, v, beta, Akk_inv, g, BT), "torch_kwargs": {},
            "triton_args": (k, v, beta, Akk_inv, g),
            "triton_kwargs": {"chunk_size": BT},
        }
    if kid in ("K5", "K6"):
        B, T, H, K_ = k.shape
        V = v.shape[-1]
        initial_state = (torch.randn(B, H, V, K_, dtype=torch.float32) * 0.05).to(device)
        initial_state_indices = torch.arange(B, dtype=torch.int32).to(device)
        k5_init_torch = initial_state.detach().clone()
        k5_init_triton = initial_state.detach().clone()
        kernels_in["K5"] = {
            "torch_args": (kg, w, u, g, k5_init_torch, initial_state_indices, BT),
            "torch_kwargs": {},
            "triton_args": (kg, w, u, g, k5_init_triton, initial_state_indices),
            "triton_kwargs": {"chunk_size": BT},
            "init_backup": initial_state.detach().clone(),
        }
    if kid == "K6":
        kernels_in["K6"] = {
            "torch_args": (q, v_new_setup, g, Aqk_merge, h_setup, scale, BT),
            "torch_kwargs": {},
            "triton_args": (q, v_new_setup, g, Aqk_merge, h_setup, scale),
            "triton_kwargs": {"chunk_size": BT},
        }
    return kernels_in


def time_kernel(kid, kernels_in, warmup, repeats):
    """测单个 kernel 的 torch/triton mean 时长 + max_diff。

    K5 in-place 改 initial_state: 每次调用前从 init_backup 复位。
    返回 dict: {torch_us, triton_us, speedup, max_diff, status}。
    """
    backup = kernels_in[kid].get("init_backup")

    def torch_fn():
        if backup is not None:
            kernels_in[kid]["torch_args"][4].copy_(backup)
        return bench._call_kernel(kid, "torch", kernels_in)

    def triton_fn():
        if backup is not None:
            kernels_in[kid]["triton_args"][4].copy_(backup)
        return bench._call_kernel(kid, "triton", kernels_in)

    torch_us = _mean_time(torch_fn, warmup, repeats)
    triton_us = _mean_time(triton_fn, warmup, repeats)

    # 精度复核: 各调用一次（K5 复位）
    if backup is not None:
        torch_out = torch_fn()
        triton_out = triton_fn()
    else:
        torch_out = bench._call_kernel(kid, "torch", kernels_in)
        triton_out = bench._call_kernel(kid, "triton", kernels_in)
    d = bench._max_diff(torch_out, triton_out)
    ok = d < bench._MAX_DIFF
    return {
        "kernel": kid,
        "torch_us": torch_us,
        "triton_us": triton_us,
        "speedup": torch_us / triton_us if triton_us > 0 else 0.0,
        "max_diff": d,
        "status": "OK" if ok else "FAIL",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="KDA 6 算子 wall-clock 计时 + 精度")
    p.add_argument("--case", default="D_KV128_H96_T16384",
                   help="case_id（默认目标 case）")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--kernel", action="append", default=None,
                   help="只测指定 kernel（可多次）")
    p.add_argument("--device", default="npu:0",
                   help="NPU 设备（默认 npu:0；由容器可见卡映射）")
    p.add_argument("--json", default=os.path.join(HERE, "time_results.json"),
                   help="输出 json 路径")
    a = p.parse_args(argv)

    if a.device:
        torch.npu.set_device(a.device)

    meta_path = os.path.join(HERE, "cases_meta.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    if a.case not in meta:
        raise SystemExit(f"[!] case 不存在: {a.case}")
    m = meta[a.case]
    B, T, H, K, V = m["B"], m["T"], m["H"], m["K"], m["V"]
    print(f"# case {a.case}  B={B} T={T} H={H} K={K} V={V}  device={a.device}")
    print(f"# 计时: warmup={a.warmup} repeats={a.repeats} (mean us, wall-clock)\n")

    torch.manual_seed(20260815)
    base = bench._gen_case_inputs(B, T, H, K)

    want = set(a.kernel) if a.kernel else None
    if want:
        # 单 kernel: 只派生到该 kernel 的前缀链，避免全链 K5 torch 开销
        kernels_in = {}
        for kid in sorted(want, key=lambda x: "123456".index(x[-1])):
            kernels_in.update(derive_prefix(kid, base, a.device))
    else:
        kernels_in = bench._derive_inputs(base, a.device)
    results = []
    total_triton = 0.0
    for km in bench.KERNEL_META:
        kid = km["id"]
        if want is not None and kid not in want:
            continue
        supports, reason = bench._kernel_supports(kid, B, T, H, K, V)
        if not supports:
            print(f"  {kid:>3} 不支持: {reason}")
            results.append({"kernel": kid, "status": "不支持", "reason": reason})
            continue
        try:
            r = time_kernel(kid, kernels_in, a.warmup, a.repeats)
        except Exception as e:
            print(f"  {kid:>3} 运行错误: {e}")
            results.append({"kernel": kid, "status": "FAIL", "error": str(e)})
            continue
        results.append(r)
        total_triton += r["triton_us"]
        print(f"  {kid:>3}  torch={r['torch_us']:9.1f}us  "
              f"triton={r['triton_us']:9.1f}us  "
              f"speedup={r['speedup']:6.1f}x  max_diff={r['max_diff']:.2e}  "
              f"{r['status']}")

    if not want:
        print(f"\n  总 triton ≈ {total_triton/1000:.2f}ms  "
              f"(0.4x H100 = 17.25ms)")

    with open(a.json, "w", encoding="utf-8") as f:
        json.dump({"case": a.case, "repeats": a.repeats, "warmup": a.warmup,
                   "results": results, "total_triton_ms": total_triton / 1000},
                  f, ensure_ascii=False, indent=2)
    print(f"\ntime_results -> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
