#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""KDA 6 算子统一 bench: 正确性（模式 A）+ msprof 分段采集（模式 B）。

数据流（每个 case，见 README.md §4）:
    g       = K1_torch(x, A_log, dt_bias)                          # [B,T,H,K]
    Aqk_d, Akk  = K2_torch(q, k, g, beta, scale)                   # [B,T,H,BT],[B,T,H,BC]
    Aqk_nd, Akk_inv = K3_torch(q, k, g, beta, Akkd=Akk, scale)     # [B,T,H,BT],[B,T,H,BT]
    w, u, kg = K4_torch(k, v, beta, A=Akk_inv, gk=g)                # 各 [B,T,H,K]
    h, v_new = K5_torch(kg, w, u, gk=g, initial_state=init.clone(), idx)
    Aqk_merge = Aqk_d + Aqk_nd
    o       = K6_torch(q, v_new, g, Aqk=Aqk_merge, h=h, scale)

对 K1..K6 各自用**完全相同的输入**（torch 链算出的中间量）分别跑 K_torch 与
K_triton，得到 max_diff。模式 B 下用 marker kernel 分段供 per_case_profile.py 切分。

用法:
  模式 A (正确性，默认):
    python3 bench.py [cases_meta.json] [--limit N] [--max-diff 1e-2]
  模式 B (msprof 采集):
    python3 bench.py --msprof [--repeats 5] [--warmup 2] [--limit N]
"""

import argparse
import csv
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
DESIGN = os.path.dirname(HERE)
# 把 6 个 op 目录的 src/ 加进 sys.path，以便 import 各 kernel 模块
for op in ("gate_chunk_cumsum", "token_parallel", "inter_solve",
           "recompute_w_u", "delta_rule_h", "gla_output"):
    sys.path.insert(0, os.path.join(DESIGN, op, "src"))
# gate_chunk_cumsum/util 下的 csvb64（本目录无 util）
sys.path.insert(0, os.path.join(DESIGN, "gate_chunk_cumsum"))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

# 6 个算子的 driver 函数
from gate_kernel import (  # noqa: E402
    RCP_LN2,
    gate_chunk_cumsum as k1_triton,
    gate_cumsum_torch as k1_torch,
)
from token_parallel_kernel import (  # noqa: E402
    token_parallel_torch as k2_torch,
    token_parallel_triton as k2_triton,
)
from inter_solve_kernel import (  # noqa: E402
    inter_solve_torch as k3_torch,
    inter_solve_triton as k3_triton,
)
from recompute_w_u_kernel import (  # noqa: E402
    recompute_w_u_torch as k4_torch,
    recompute_w_u_triton as k4_triton,
)
from delta_rule_h_kernel import (  # noqa: E402
    delta_rule_h_torch as k5_torch,
    delta_rule_h_triton as k5_triton,
)
from gla_output_kernel import (  # noqa: E402
    gla_output_torch as k6_torch,
    gla_output_kernel as k6_triton,
)

_BT = 64
_BC = 16
_MAX_DIFF = 1e-2
# NPU coreDim 上限: triton-ascend 把 3D/2D grid 展平为 1D 后, 总 grid 数
# 不能超过 65535 (rtKernelLaunch 报 ERR00100 "value 65536 for parameter
# coreDim is invalid")。
_NPU_CORE_DIM_MAX = 65535


def _cdiv(a, b):
    """向上取整的整数除法。"""
    return -(a // -b)

# 6 kernel 的 triton op 名（供 per_case_profile.py 按 op_summary 行序分段）
KERNEL_META = [
    {"id": "K1", "name": "gate_chunk_cumsum",  "triton_op": "_gate_cumsum_kernel"},
    {"id": "K2", "name": "token_parallel",     "triton_op": "_token_parallel_kernel"},
    {"id": "K3", "name": "inter_solve",        "triton_op": "_inter_solve_kernel"},
    {"id": "K4", "name": "recompute_w_u",     "triton_op": "_recompute_w_u_kernel"},
    {"id": "K5", "name": "delta_rule_h",      "triton_op": "_delta_rule_h_kernel"},
    {"id": "K6", "name": "gla_output",        "triton_op": "chunk_gla_fwd_kernel_o"},
]


# ─── marker kernel (msprof 分段用) ──────────────────────────────────────────


@triton.jit
def _kda_bench_marker(dummy):
    """1 元素 store 的微 kernel，op_summary 中作 (case,kernel) 分界。"""
    tl.store(dummy + tl.arange(0, 1), tl.zeros((1,), dtype=tl.float32))


def _emit_marker(device):
    """发一个 marker kernel（1 元素 store，微秒级），用于 op_summary 分段。"""
    dummy = torch.zeros(1, dtype=torch.float32, device=device)
    _kda_bench_marker[(1,)](dummy)
    torch.npu.synchronize()


# ─── 用例张量生成（固定种子，与 gen_cases.py 分布一致）──────────────────────


def _gen_case_inputs(B, T, H, K):
    """按固定种子即时生成 case 的共享基础张量（fp32, CPU）。

    与 gen_cases.py::_mk_shared_inputs 分布一致:
      x = randn*0.5 - 2.0; A_log = randn*0.1; dt_bias = randn*0.1;
      q,k = L2-normalize(randn); v = randn*0.1; beta = sigmoid(rand);
      scale = 1/sqrt(K).
    全部在 CPU 生成，由 _derive_inputs 负责 .to(device) 搬上 NPU。
    """
    V = K
    x = torch.randn(B, T, H, K, dtype=torch.float32) * 0.5 - 2.0
    A_log = torch.randn(H, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(H * K, dtype=torch.float32) * 0.1
    q = torch.nn.functional.normalize(
        torch.randn(B, T, H, K, dtype=torch.float32), dim=-1)
    k = torch.nn.functional.normalize(
        torch.randn(B, T, H, K, dtype=torch.float32), dim=-1)
    v = torch.randn(B, T, H, V, dtype=torch.float32) * 0.1
    beta = torch.rand(B, T, H, dtype=torch.float32).sigmoid()
    scale = 1.0 / (float(K) ** 0.5)
    return {
        "x": x, "A_log": A_log, "dt_bias": dt_bias,
        "q": q, "k": k, "v": v, "beta": beta,
        "scale": scale,
    }


# ─── 派生中间量（torch_npu 链，setup 只为拿到 K1-K6 各自的输入）─────────────


def _derive_inputs(base, device):
    """按 §4.1 数据流用 torch_npu 链算出 K1-K6 各自所需的输入（都在 NPU）。

    返回 dict: {kernel_id: {"torch_args": ..., "triton_args": ...}, ...}。
    torch 链本身不被计时（setup 阶段，会出现在 op_summary 首个 marker 之前）。
    """
    x = base["x"].to(device)
    A_log = base["A_log"].to(device)
    dt_bias = base["dt_bias"].to(device)
    q = base["q"].to(device)
    k = base["k"].to(device)
    v = base["v"].to(device)
    beta = base["beta"].to(device)
    scale = base["scale"]
    BT = _BT

    # K1: g = gate(x, A_log, dt_bias) — torch_npu 链
    g = k1_torch(x, A_log, dt_bias=dt_bias, chunk_size=BT, scale=RCP_LN2)
    # K2: Aqk_d, Akk = token_parallel(q, k, gk=g, beta, scale)
    Aqk_d, Akk = k2_torch(q, k, g, beta, scale, chunk_size=BT, sub_chunk_size=_BC)
    # K3: Aqk_nd, Akk_inv = inter_solve(q, k, g, beta, Akkd=Akk, scale)
    Aqk_nd, Akk_inv = k3_torch(
        q, k, g, beta, Akkd=Akk, scale=scale,
        chunk_size=BT, sub_chunk_size=_BC,
    )
    # K4: w, u, kg = recompute_w_u(k, v, beta, A=Akk_inv, gk=g)
    w, u, kg = k4_torch(k, v, beta, A=Akk_inv, gk=g, chunk_size=BT)
    # K5 输入: kg, w, u, gk=g, initial_state (per-batch), indices=arange(B)
    B, T, H, K_ = k.shape
    V = v.shape[-1]
    # initial_state/indices 在 CPU 创建再搬 NPU，避免 NPU randn/arange 对
    # int32 indices 产生不稳定结果（实测 B=2/4/8 时 NPU arange 给出垃圾值）。
    initial_state = (torch.randn(B, H, V, K_, dtype=torch.float32) * 0.05).to(device)
    initial_state_indices = torch.arange(B, dtype=torch.int32).to(device)
    # K6 输入: q, v_new, g, Aqk_merge, h, scale — 需先算 K5 得到 h, v_new
    # 用 torch_npu 算一次 setup 用值（K6 的 v_new/h 输入由 setup 阶段产生）。
    # 保留独立的 initial_state 备份供 K5 每次调用前复位（in-place 更新）；
    # 这里多 clone 几份避免跨 case 共享同一 NPU 缓冲。
    init_backup = initial_state.detach().clone()
    h_setup, v_new_setup = k5_torch(
        kg, w, u, gk=g, initial_state=initial_state.detach().clone(),
        initial_state_indices=initial_state_indices, chunk_size=BT,
    )
    Aqk_merge = Aqk_d + Aqk_nd

    # 组装每个 kernel 的 torch/triton 调用参数（与 driver 签名一致）
    # K5 的 initial_state 单独存一份，避免与 setup/init_backup 共享内存
    k5_init_torch = initial_state.detach().clone()
    k5_init_triton = initial_state.detach().clone()
    kernels_in = {
        "K1": {
            "torch_args": (x, A_log, dt_bias, BT, RCP_LN2),
            "torch_kwargs": {},
            "triton_args": (x, A_log, dt_bias, BT, RCP_LN2),
            "triton_kwargs": {},
        },
        "K2": {
            "torch_args": (q, k, g, beta, scale, BT, _BC),
            "torch_kwargs": {},
            "triton_args": (q, k, g, beta, scale),
            "triton_kwargs": {"chunk_size": BT, "sub_chunk_size": _BC},
        },
        "K3": {
            "torch_args": (q, k, g, beta, Akk, scale, BT, _BC),
            "torch_kwargs": {},
            "triton_args": (q, k, g, beta, Akk, scale),
            "triton_kwargs": {"chunk_size": BT, "sub_chunk_size": _BC},
        },
        "K4": {
            "torch_args": (k, v, beta, Akk_inv, g, BT),
            "torch_kwargs": {},
            "triton_args": (k, v, beta, Akk_inv, g),
            "triton_kwargs": {"chunk_size": BT},
        },
        "K5": {
            "torch_args": (kg, w, u, g, k5_init_torch, initial_state_indices, BT),
            "torch_kwargs": {},
            "triton_args": (kg, w, u, g, k5_init_triton, initial_state_indices),
            "triton_kwargs": {"chunk_size": BT},
            "init_backup": init_backup,
        },
        "K6": {
            "torch_args": (q, v_new_setup, g, Aqk_merge, h_setup, scale, BT),
            "torch_kwargs": {},
            "triton_args": (q, v_new_setup, g, Aqk_merge, h_setup, scale),
            "triton_kwargs": {"chunk_size": BT},
        },
    }
    return kernels_in


# ─── kernel 调用 dispatcher ──────────────────────────────────────────────


def _call_kernel(kernel_id, impl, kernels_in):
    """调用某 kernel 的 torch 或 triton 实现，返回输出（用于精度对比）。"""
    ki = kernels_in[kernel_id]
    if impl == "torch":
        fn = _TORCH_FNS[kernel_id]
        args = ki["torch_args"]
        kwargs = ki["torch_kwargs"]
    else:
        fn = _TRITON_FNS[kernel_id]
        args = ki["triton_args"]
        kwargs = ki["triton_kwargs"]
    return fn(*args, **kwargs)


_TRITON_FNS = {
    "K1": k1_triton, "K2": k2_triton, "K3": k3_triton,
    "K4": k4_triton, "K5": k5_triton, "K6": k6_triton,
}
_TORCH_FNS = {
    "K1": k1_torch, "K2": k2_torch, "K3": k3_torch,
    "K4": k4_torch, "K5": k5_torch, "K6": k6_torch,
}


def _kernel_grid_size(kernel_id, B, T, H, K, V):
    """返回 triton kernel 的展平 grid 大小（各维乘积）。

    用于检查是否超过 NPU coreDim 上限 65535。grid 形状取自各 kernel driver:
      K1: (cdiv(K,32), cdiv(T,64), B*H)
      K2: (B*T, H)
      K3: (cdiv(T,64), B*H)
      K4: (cdiv(T,64), B*H)
      K5: (cdiv(V,32), B*H)
      K6: (cdiv(V,32), cdiv(T,64), B*H)
    """
    BT = _BT
    if kernel_id == "K1":
        return _cdiv(K, 128) * _cdiv(T, BT) * (B * H)  # BS=128 (见 gate_kernel.py OPTIMIZATION_LOG.md 第二轮优化)
    if kernel_id == "K2":
        return _cdiv(T, BT) * (B * H)  # chunked grid: (B*cdiv(T,BT), H)
    if kernel_id in ("K3", "K4"):
        return _cdiv(T, BT) * (B * H)
    if kernel_id == "K5":
        return _cdiv(V, 32) * (B * H)
    if kernel_id == "K6":
        return _cdiv(V, 128) * _cdiv(T, BT) * (B * H)  # BV=128
    return 0


def _kernel_supports(kernel_id, B, T, H, K, V):
    """检查 triton kernel 是否支持该 case shape（返回 (supports, reason)）。

    已知约束:
      * 所有 kernel: triton-ascend 把 3D/2D grid 展平为 1D, 总 grid 数
        不能超过 NPU coreDim 上限 65535 (rtKernelLaunch 报 ERR00100
        "value 65536 for parameter coreDim is invalid"), 否则 kernel 启动
        失败并可能污染 NPU 设备状态 (导致后续 kernel/torch_npu 算子也失败)。
      * K5 (delta_rule_h): 要求 K==V 且 K≤256 (K=128 已支持; K=64 走 flat 1D store,
        K≠64 走 2D store)
      * K=32: K2/K3/K4 tl.dot 在 BK=32 时不稳定; K1/K6 大 T 下精度不足
    """
    # 1) 展平 grid 上限 (所有 kernel 通用)
    grid_size = _kernel_grid_size(kernel_id, B, T, H, K, V)
    if grid_size > _NPU_CORE_DIM_MAX:
        return False, (
            f"{kernel_id} grid(flattened)={grid_size} 超过 NPU coreDim 上限 "
            f"{_NPU_CORE_DIM_MAX}"
        )
    # 2) K5 K=V 约束 (已修复 K=128, 支持 K≤256)
    if kernel_id == "K5":
        if K != V:
            return False, f"K5 要求 K==V, 实际 K={K},V={V}"
        if K > 256:
            return False, f"K5 仅支持 K≤256, 实际 K={K}"
    # 3) K=32 精度/稳定性约束
    if K == 32:
        if kernel_id in ("K2", "K3", "K4"):
            return False, f"{kernel_id} triton 不支持 K=32 (BK=32 时 tl.dot 不稳定)"
        if kernel_id in ("K1", "K6") and T > 256:
            return False, f"{kernel_id} triton K=32 仅小 T (≤256) 验证通过, T={T}"
    return True, ""


def _max_diff(a, b):
    """fp32 max diff（CPU 上比）。"""
    if isinstance(a, tuple):
        assert isinstance(b, tuple)
        return max(_max_diff(x, y) for x, y in zip(a, b))
    if a is None or b is None:
        return 0.0
    return (a.float().cpu() - b.float().cpu()).abs().max().item()


def _filter_cases(cases, group=None, start=0, limit=0):
    """按 group/起始/数量过滤 case 列表。"""
    if group is not None:
        cases = [(cid, m) for cid, m in cases if m["group"] == group]
    if start > 0:
        cases = cases[start:]
    if limit > 0:
        cases = cases[:limit]
    return cases


# ─── 模式 A: 正确性 ────────────────────────────────────────────────────────


def run_correctness(meta_path, limit=0, max_diff=_MAX_DIFF, group=None,
                    start=0, verbose=True, device="npu"):
    """模式 A: 逐 case 跑 6 kernel 的 torch/triton，写 correctness.csv。"""
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        print("[!] 未检测到可用 NPU — 正确性模式需要 NPU（torch_npu 与 triton 两版都需 NPU）")
        return 2

    cases = list(meta.items())
    cases = _filter_cases(cases, group=group, start=start, limit=limit)

    out_csv = os.path.join(HERE, "correctness.csv")
    rows = []
    n_pass = 0
    for cid, m in cases:
        B, T, H, K, V = m["B"], m["T"], m["H"], m["K"], m["V"]
        torch.manual_seed(20260815)
        base = _gen_case_inputs(B, T, H, K)
        try:
            kernels_in = _derive_inputs(base, device)
        except Exception as e:
            print(f"[{cid}] setup 失败: {e}")
            for km in KERNEL_META:
                rows.append({
                    "case_id": cid, "B": B, "T": T, "H": H, "K": K, "V": V,
                    "kernel": km["id"], "max_diff": f"SETUP_ERR: {e}",
                    "status": "FAIL",
                })
            continue

        for km in KERNEL_META:
            kid = km["id"]
            supports, reason = _kernel_supports(kid, B, T, H, K, V)
            if not supports:
                rows.append({
                    "case_id": cid, "B": B, "T": T, "H": H, "K": K, "V": V,
                    "kernel": kid, "max_diff": reason, "status": "不支持",
                })
                if verbose:
                    print(f"[{cid:>24}] {kid} 不支持: {reason}")
                continue
            try:
                # K5 会 in-place 改 initial_state: 两版各自把 init 复位到备份值再调用
                if kid == "K5":
                    backup = kernels_in[kid]["init_backup"]
                    init_torch = kernels_in[kid]["torch_args"][4]
                    init_triton = kernels_in[kid]["triton_args"][4]
                    init_torch.copy_(backup)
                    out_torch = _call_kernel(kid, "torch", kernels_in)
                    init_triton.copy_(backup)
                    out_tri = _call_kernel(kid, "triton", kernels_in)
                    # 调用后复位，避免后续误用被 in-place 改过的 init
                    init_torch.copy_(backup)
                    init_triton.copy_(backup)
                else:
                    out_torch = _call_kernel(kid, "torch", kernels_in)
                    out_tri = _call_kernel(kid, "triton", kernels_in)
                d = _max_diff(out_torch, out_tri)
                ok = d < max_diff
                rows.append({
                    "case_id": cid, "B": B, "T": T, "H": H, "K": K, "V": V,
                    "kernel": kid, "max_diff": f"{d:.6e}", "status": "OK" if ok else "FAIL",
                })
                if verbose:
                    print(f"[{cid:>24}] {kid} max_diff={d:.3e} {'OK' if ok else 'FAIL'}")
                if ok:
                    n_pass += 1
            except Exception as e:
                rows.append({
                    "case_id": cid, "B": B, "T": T, "H": H, "K": K, "V": V,
                    "kernel": kid, "max_diff": f"RUN_ERR: {e}", "status": "FAIL",
                })
                if verbose:
                    print(f"[{cid:>24}] {kid} 运行错误: {e}")

    # 写 correctness.csv
    with open(out_csv, "w", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "case_id", "B", "T", "H", "K", "V", "kernel", "max_diff", "status"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n正确性结果 -> {out_csv}")
    print(f"PASS: {n_pass}/{len(rows)} rows OK")
    return 0 if n_pass == len(rows) else 1


# ─── 模式 B: msprof 采集（marker 分段）─────────────────────────────────────


def run_msprof(meta_path, repeats=5, warmup=2, limit=0, group=None, start=0,
               device="npu"):
    """模式 B: 逐 case 派生输入 → 对每个支持的 kernel 在 marker 分界内跑
    K_torch N 次 + K_triton N 次（不计时）。写 profile_meta.json。"""
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        print("[!] 未检测到可用 NPU — msprof 模式需要 NPU")
        return 2

    cases = list(meta.items())
    cases = _filter_cases(cases, group=group, start=start, limit=limit)

    profile_meta = []
    for cid, m in cases:
        B, T, H, K, V = m["B"], m["T"], m["H"], m["K"], m["V"]
        torch.manual_seed(20260815)
        base = _gen_case_inputs(B, T, H, K)
        try:
            kernels_in = _derive_inputs(base, device)
        except Exception as e:
            print(f"[{cid}] setup 失败: {e}")
            continue

        for km in KERNEL_META:
            kid = km["id"]
            supports, reason = _kernel_supports(kid, B, T, H, K, V)
            if not supports:
                profile_meta.append({
                    "case_id": cid, "kernel": kid, "triton_op": km["triton_op"],
                    "repeats": 0, "skipped": True, "reason": reason,
                })
                continue

            # ── marker: case,kernel torch 段开始 ──
            _emit_marker(device)
            # warmup + repeats: torch
            try:
                if kid == "K5":
                    backup = kernels_in[kid]["init_backup"]
                    init_torch = kernels_in[kid]["torch_args"][4]
                    init_triton = kernels_in[kid]["triton_args"][4]
                    for _ in range(warmup):
                        init_torch.copy_(backup)
                        _call_kernel(kid, "torch", kernels_in)
                    for _ in range(repeats):
                        init_torch.copy_(backup)
                        _call_kernel(kid, "torch", kernels_in)
                    # ── marker: torch 段结束 / triton 段开始 ──
                    _emit_marker(device)
                    for _ in range(warmup):
                        init_triton.copy_(backup)
                        _call_kernel(kid, "triton", kernels_in)
                    for _ in range(repeats):
                        init_triton.copy_(backup)
                        _call_kernel(kid, "triton", kernels_in)
                    # 收尾复位
                    init_torch.copy_(backup)
                    init_triton.copy_(backup)
                else:
                    for _ in range(warmup):
                        _call_kernel(kid, "torch", kernels_in)
                    for _ in range(repeats):
                        _call_kernel(kid, "torch", kernels_in)
                    # ── marker: torch 段结束 / triton 段开始 ──
                    _emit_marker(device)
                    for _ in range(warmup):
                        _call_kernel(kid, "triton", kernels_in)
                    for _ in range(repeats):
                        _call_kernel(kid, "triton", kernels_in)
            except Exception as e:
                print(f"[{cid}] {kid} 运行错误: {e}")
                profile_meta.append({
                    "case_id": cid, "kernel": kid, "triton_op": km["triton_op"],
                    "repeats": 0, "skipped": True, "reason": f"RUN_ERR: {e}",
                })
                _emit_marker(device)
                continue
            # ── marker: case,kernel 段结束 ──
            _emit_marker(device)
            profile_meta.append({
                "case_id": cid, "kernel": kid, "triton_op": km["triton_op"],
                "repeats": repeats, "skipped": False, "reason": "",
            })
            print(f"[{cid:>24}] {kid} torch×{repeats} + triton×{repeats} done")

    # 写 profile_meta.json
    out_json = os.path.join(HERE, "profile_meta.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(profile_meta, f, ensure_ascii=False, indent=2)
    print(f"\nprofile_meta -> {out_json}")
    print(f"segments: {len(profile_meta)}")
    return 0


# ─── main ────────────────────────────────────────────────────────────────


def main(argv=None):
    HERE = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="KDA 6 算子统一 bench")
    p.add_argument("meta", nargs="?", default=os.path.join(HERE, "cases_meta.json"),
                   help="cases_meta.json 路径（默认 %(default)s）")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 个 case（0=全部）")
    p.add_argument("--max-diff", type=float, default=_MAX_DIFF, help="精度上限")
    p.add_argument("--msprof", action="store_true", help="模式 B: msprof 采集")
    p.add_argument("--repeats", type=int, default=5, help="每个 kernel 每实现重复次数")
    p.add_argument("--warmup", type=int, default=2, help="预热次数")
    p.add_argument("--group", default=None, help="只跑指定 group (A/B/C/D)")
    p.add_argument("--start", type=int, default=0, help="从第 N 个 case 开始")
    p.add_argument("--device", default=None,
                   help="NPU 设备（默认 npu:0 当前卡）；如 npu:1。会先 torch.npu.set_device")
    a = p.parse_args(argv)

    device = a.device or "npu"
    if a.device:
        torch.npu.set_device(a.device)

    if a.msprof:
        return run_msprof(a.meta, repeats=a.repeats, warmup=a.warmup,
                          limit=a.limit, group=a.group, start=a.start,
                          device=device)
    return run_correctness(a.meta, limit=a.limit, max_diff=a.max_diff,
                            group=a.group, start=a.start, device=device)


if __name__ == "__main__":
    sys.exit(main())
