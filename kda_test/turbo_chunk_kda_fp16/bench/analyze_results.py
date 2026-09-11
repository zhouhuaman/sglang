#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""解析 results.csv，输出:
  1. 加速比矩阵（行=case，列=K1..K6，格=speedup）
  2. 每 kernel 汇总（支持 case 数、不支持 case 数、平均/中位 speedup、随 T 变化）
  3. 6 kernel 相对时间（指定代表 case 下 torch_us / triton_us 排序占比）

用法:
    python3 analyze_results.py [--csv results.csv] [--pivot-case A_B1_H8_T16384]
"""
import argparse
import csv
import os
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL_ORDER = ["gate_chunk_cumsum", "token_parallel", "inter_solve", "recompute_w_u", "delta_rule_h", "gla_output"]
# per_case_profile.py 的 results.csv 用 kernel id (K1..K6) 而非全名，加载时映射
KID_TO_NAME = {f"K{i + 1}": name for i, name in enumerate(KERNEL_ORDER)}


def _load_results(csv_path: str):
    """读 results.csv，返回 list[dict]。"""
    if not os.path.exists(csv_path):
        raise SystemExit(f"[!] {csv_path} 不存在；请先跑 per_case_profile.py")
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["kernel"] = KID_TO_NAME.get(r["kernel"], r["kernel"])
    return rows


def _is_num(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _speedup_matrix(rows):
    """输出 case × kernel 加速比矩阵（文本表格）。"""
    # case 顺序: 按 T 升序 → H → B → case_id（稳定排序）
    def _sort_key(r):
        try:
            return (int(r["T"]), int(r["H"]), int(r["B"]), r["case_id"])
        except (ValueError, KeyError):
            return (0, 0, 0, r.get("case_id", ""))

    sorted_rows = sorted(rows, key=_sort_key)
    # 按 case_id 聚合（同 case 的 6 个 kernel 行）
    by_case = defaultdict(dict)
    for r in sorted_rows:
        by_case[r["case_id"]][r["kernel"]] = r

    # 表头
    print("\n== 加速比矩阵 (speedup = torch_us / triton_us) ==")
    print(f"{'case_id':>26} " + " ".join(f"{k:>9}" for k in KERNEL_ORDER))
    print("-" * (27 + 10 * len(KERNEL_ORDER)))
    for cid, krows in by_case.items():
        cells = []
        for k in KERNEL_ORDER:
            r = krows.get(k)
            if r is None:
                cells.append("".rjust(9))
            elif _is_num(r.get("speedup", "")):
                sp = float(r["speedup"])
                cells.append(f"{sp:>8.2f}x")
            else:
                cells.append("    -   ".rjust(9))
        print(f"{cid:>26} " + " ".join(cells))


def _per_kernel_summary(rows):
    """每 kernel 汇总: 支持/不支持数、平均/中位 speedup、T→speedup 趋势。"""
    print("\n== 每 kernel 汇总 ==")
    print(f"{'kernel':>6} {'supported':>10} {'skipped':>8} "
          f"{'avg_sp':>8} {'med_sp':>8} {'min_sp':>8} {'max_sp':>8}")
    print("-" * 70)
    for k in KERNEL_ORDER:
        krows = [r for r in rows if r["kernel"] == k]
        n_sup = sum(1 for r in krows
                    if _is_num(r.get("speedup", "")))
        n_skip = len(krows) - n_sup
        sps = [float(r["speedup"]) for r in krows
               if _is_num(r.get("speedup", ""))]
        if sps:
            avg = statistics.mean(sps)
            med = statistics.median(sps)
            mn = min(sps)
            mx = max(sps)
            print(f"{k:>6} {n_sup:>10} {n_skip:>8} "
                  f"{avg:>7.2f}x {med:>7.2f}x {mn:>7.2f}x {mx:>7.2f}x")
        else:
            print(f"{k:>6} {n_sup:>10} {n_skip:>8}  "
                  f"{'N/A':>7} {'N/A':>7} {'N/A':>7} {'N/A':>7}")


def _relative_time(rows, pivot_case=None):
    """6 kernel 在代表 case 下的相对时间表。"""
    # 选代表 case: 优先 --pivot-case；否则取各 T 档（2 的幂）第一个 K=64 case
    by_case = defaultdict(list)
    for r in rows:
        by_case[r["case_id"]].append(r)

    if pivot_case and pivot_case in by_case:
        cases_to_show = [pivot_case]
    else:
        # 自动取各 T 档（2 的幂）一个代表 case: T ∈ {1024, 4096, 16384, 65536}
        target_ts = [1024, 4096, 16384, 65536]
        seen_ts = set()
        cases_to_show = []
        for cid, rs in by_case.items():
            try:
                t = int(rs[0]["T"])
            except (ValueError, KeyError):
                continue
            if t in target_ts and t not in seen_ts:
                cases_to_show.append(cid)
                seen_ts.add(t)
        cases_to_show = sorted(cases_to_show,
                               key=lambda c: int(by_case[c][0]["T"]))

    if not cases_to_show:
        print("\n[!] 没有找到代表 case 可做相对时间分析")
        return

    print("\n== 6 kernel 相对时间（代表 case）==")
    for cid in cases_to_show:
        rs = by_case[cid]
        # 取出该 case 下 6 kernel 的 torch_us / triton_us
        torch_vals = {}
        triton_vals = {}
        for r in rs:
            k = r["kernel"]
            if _is_num(r.get("torch_us", "")):
                torch_vals[k] = float(r["torch_us"])
            if _is_num(r.get("triton_us", "")):
                triton_vals[k] = float(r["triton_us"])
        # 表头
        print(f"\n  case={cid} (T={rs[0].get('T','?')}, "
              f"B={rs[0].get('B','?')}, H={rs[0].get('H','?')})")
        print(f"  {'kernel':>6} {'torch_us':>12} {'tri_us':>10} "
              f"{'torch%':>8} {'tri%':>8} {'speedup':>8}")
        tot_t = sum(torch_vals.values()) or 1.0
        tot_r = sum(triton_vals.values()) or 1.0
        for k in KERNEL_ORDER:
            tv = torch_vals.get(k)
            rv = triton_vals.get(k)
            if tv is None or rv is None:
                print(f"  {k:>6} {'(unsupported)':>12} {'':>10} "
                      f"{'':>8} {'':>8} {'':>8}")
                continue
            tp = tv / tot_t * 100
            rp = rv / tot_r * 100
            sp = tv / rv if rv > 0 else float("nan")
            print(f"  {k:>6} {tv:>11.2f}us {rv:>9.2f}us "
                  f"{tp:>7.1f}% {rp:>7.1f}% {sp:>7.2f}x")
        print(f"  {'total':>6} {tot_t:>11.2f}us {tot_r:>9.2f}us "
              f"{100.0:>7.1f}% {100.0:>7.1f}%")


def main(argv=None):
    p = argparse.ArgumentParser(
        description="解析 results.csv: 加速比矩阵 / kernel 汇总 / 相对时间")
    p.add_argument("--csv", default=os.path.join(HERE, "results.csv"),
                   help="results.csv 路径（默认 %(default)s）")
    p.add_argument("--pivot-case", help="指定相对时间分析的代表 case_id")
    a = p.parse_args(argv)

    rows = _load_results(a.csv)
    print(f"# 读取 {len(rows)} 行 (来自 {a.csv})")
    _speedup_matrix(rows)
    _per_kernel_summary(rows)
    _relative_time(rows, pivot_case=a.pivot_case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
