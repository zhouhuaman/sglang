#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""统一测试用例表生成器（KDA 6 算子共用）。

构造 106 个 case（T 覆盖 1k–128k）:
  * A 组（T 主扫描，2 的幂）: B∈{1,2,4}, H∈{2,4,8}, T∈{1024..131072} 共 72
  * B 组（T 非 2 幂/边界）  : B=1, H=8, 20 个非对齐 T                       共 20
  * C 组（多 batch 放大）   : B=8, H∈{4,8}, T∈{4096..32768}                共 8
  * D 组（K=V 扩展 + 目标 case）: K=V=128 {1024,16384}×H∈{2,8} (4) +
    K=V=32 {4096,H=4} (1) + K=V=128 H=96 T=16384（目标 case）             共 6

K=V=128 下 K5 已支持（K≤256，K=128 走 2D store）；D 组唯一"不支持演示"为
K=V=32 大 T（K1/K6 仅小 T、K2/K3/K4 BK=32 时 tl.dot 不稳定）。

只写 cases_meta.json（case 表: id→{B,T,H,K,V,group,desc}），**不写张量数据**。
张量由 bench.py 按固定种子即时生成（见 bench.py::_gen_case_inputs），避免
T=131072/B=8/H=8 的多 GB 级 CSV 文件。

用法:
  python3 gen_cases.py                      # 默认 cases_meta.json
  python3 gen_cases.py --meta m.json
  python3 gen_cases.py --limit 5            # 只生成前 5 个 case (smoke 用)

宿主机可直接跑（不依赖 torch/NPU）。
"""

import argparse
import json
import os
import sys


def _case_table():
    """(case_id, B, T, H, K, V, group, desc) — 105 行。"""
    cases = []

    # A 组：T 主扫描，2 的幂
    T_powers = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    for B in (1, 2, 4):
        for H in (2, 4, 8):
            for T in T_powers:
                cases.append((
                    f"A_B{B}_H{H}_T{T}", B, T, H, 64, 64, "A",
                    f"T 主扫描 2 幂: B={B},H={H},T={T}",
                ))

    # B 组：T 非 2 幂 / 边界（B=1, H=8）
    B_vals = [
        1023, 1025, 2047, 2049, 4095, 4097, 8191, 8193,
        16383, 16385, 32767, 32769, 65535, 65537,
        131071, 1536, 6144, 24576, 98304, 100000,
    ]
    for T in B_vals:
        cases.append((
            f"B_T{T}", 1, T, 8, 64, 64, "B",
            f"T 非 2 幂/边界: T={T}",
        ))

    # C 组：多 batch 放大（B=8, H∈{4,8}, T∈{4096..32768}）
    for H in (4, 8):
        for T in (4096, 8192, 16384, 32768):
            cases.append((
                f"C_B8_H{H}_T{T}", 8, T, H, 64, 64, "C",
                f"多 batch 放大: B=8,H={H},T={T}",
            ))

    # D 组：K=V 扩展 + 目标 case（K5 支持 K=V≤256；K=V=128 已支持）
    # D1-D4: K=V=128, T∈{1024,16384}, H∈{2,8}
    for H in (2, 8):
        for T in (1024, 16384):
            cases.append((
                f"D_KV128_H{H}_T{T}", 1, T, H, 128, 128, "D",
                f"K=V=128: H={H},T={T}",
            ))
    # D5: K=V=32, T=4096, H=4（唯一"不支持演示": K1/K6 仅小T, K2/K3/K4 BK=32 不稳定）
    cases.append((
        "D_KV32_H4_T4096", 1, 4096, 4, 32, 32, "D",
        "K=V=32 大 T (K1/K6 仅小T, K2/K3/K4 BK=32 不稳定 → 不支持): H=4,T=4096",
    ))
    # D6: 目标 case（放在最后，cases_meta.json 中索引 105）
    cases.append((
        "D_KV128_H96_T16384", 1, 16384, 96, 128, 128, "D",
        "K=V=128 (目标 case): H=96,T=16384",
    ))

    return cases


def main(argv=None):
    HERE = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="KDA 6 算子统一测试用例表生成")
    p.add_argument(
        "--meta", default=os.path.join(HERE, "cases_meta.json"),
        help="输出 meta JSON 路径（默认 %(default)s）",
    )
    p.add_argument("--limit", type=int, default=0, help="只生成前 N 个 case（0=全部）")
    a = p.parse_args(argv)

    table = _case_table()
    if a.limit > 0:
        table = table[:a.limit]

    meta = {}
    for cid, B, T, H, K, V, group, desc in table:
        meta[cid] = {
            "B": B, "T": T, "H": H, "K": K, "V": V,
            "group": group, "desc": desc,
        }

    os.makedirs(os.path.dirname(os.path.abspath(a.meta)), exist_ok=True)
    with open(a.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 统计
    by_group = {}
    for m in meta.values():
        by_group[m["group"]] = by_group.get(m["group"], 0) + 1
    group_str = ", ".join(f"{g}={n}" for g, n in sorted(by_group.items()))
    print(f"wrote {len(meta)} cases -> {a.meta}")
    print(f"groups: {group_str}")
    print(f"T range: [{min(m['T'] for m in meta.values())}, "
          f"{max(m['T'] for m in meta.values())}]")
    k5_ok = sum(1 for m in meta.values() if m['K'] == m['V'] and m['K'] <= 256)
    print(f"K5 支持 (K=V≤256): {k5_ok}/{len(meta)}")
    print(f"K=V=64: {sum(1 for m in meta.values() if m['K']==64)}")
    print(f"K=V=128: {sum(1 for m in meta.values() if m['K']==128)}")


if __name__ == "__main__":
    main()
