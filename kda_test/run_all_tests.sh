#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# KDA turbo 三形态（fp32 / fp16 / bf16）一键测试。
#
#   默认：全量 106 case 全链正确性 + 6 个单算子正确性；加 --perf 附带 msprof 性能。
#
# 用法（在 kda_test/ 下）:
#   bash run_all_tests.sh                                  # 三棵树，全量 106 case 正确性
#   bash run_all_tests.sh --dtype fp16                     # 只测 fp16
#   bash run_all_tests.sh --case 105                       # 只测索引 105
#   bash run_all_tests.sh --case D_KV128_H96_T16384        # 按名字选 case
#   bash run_all_tests.sh --case 105,7,23 --dtype fp32     # 多 case + 单形态
#   bash run_all_tests.sh --target                         # = --case 105 的快捷方式
#   bash run_all_tests.sh --case 105 --perf                # 正确性 + msprof 性能
#   CARD=3 bash run_all_tests.sh                           # 指定物理卡（默认 5）
#
# 参数:
#   --dtype fp32|fp16|bf16|all   选形态（默认 all = 三棵树全跑）
#   --case  all|索引|名字|逗号列表  选 case（默认 all = 106 条全跑）
#   --target                     等价于 --case 105（目标 case，106 条中末位）
#   --perf                       附带 msprof 性能采集（全链 ② + 单算子隔离 ④）
#
# 关键点:
#   * 必须用交付口径的 triton-ascend 3.2.1（conda autotriton 的 python3），
#     默认 /usr/local/python3.11.10 是 3.2.2+dev，数字不可混用。
#   * 单算子段的 test.py 只能测目标 case（--target）或全 shape 扫描（无参、很慢），
#     不支持任意 case，故 ③/④ 固定为目标 case，与 --case 无关。
#   * run_cpu.sh 自带 mkdir -p 与 CANN 环境；裸 msprof 两者都要自己做。
#   * run_cpu.sh 的 VISIBLE_DEVICES 默认 1,2,3,4,5,6,7（屏蔽坏卡 0），会覆盖
#     env.sh 设的卡，故此处显式传 VISIBLE_DEVICES=$CARD。

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CARD="${CARD:-5}"
DTYPE="all"
CASE_SPEC="all"
WITH_PERF=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dtype)  DTYPE="${2:?--dtype 需要取值}"; shift 2 ;;
    --case)   CASE_SPEC="${2:?--case 需要取值}"; shift 2 ;;
    --target) CASE_SPEC=105; shift ;;
    --perf)   WITH_PERF=1; shift ;;
    -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "未知参数: $1（-h 看用法）" >&2; exit 2 ;;
  esac
done

# 交付口径工具链（必须在解析 case / 跑 python 之前）
export PATH=/data/anaconda3/envs/autotriton/bin:$PATH

case "$DTYPE" in
  fp32) TREES=turbo_chunk_kda ;;
  fp16) TREES=turbo_chunk_kda_fp16 ;;
  bf16) TREES=turbo_chunk_kda_bf16 ;;
  all)  TREES="turbo_chunk_kda turbo_chunk_kda_fp16 turbo_chunk_kda_bf16" ;;
  *) echo "--dtype 只能是 fp32/fp16/bf16/all，收到: $DTYPE" >&2; exit 2 ;;
esac

META="$ROOT/turbo_chunk_kda/bench/cases_meta.json"
[ -f "$META" ] || { echo "缺 $META（先在 turbo_chunk_kda/bench 下跑一次 gen_cases.py）" >&2; exit 1; }

# --case 解析：索引 / 名字（可前缀匹配）/ 逗号列表 / all → 升序索引串
CASES="$(python3 - "$META" "$CASE_SPEC" <<'PY'
import json, sys
names = list(json.load(open(sys.argv[1])))
spec = sys.argv[2].strip()
if spec in ("", "all"):
    print(" ".join(map(str, range(len(names))))); raise SystemExit
out = []
for tok in spec.replace(" ", "").split(","):
    if not tok:
        continue
    if tok.isdigit():
        i = int(tok)
        if not 0 <= i < len(names):
            sys.exit(f"case 索引越界: {tok}（合法范围 0..{len(names) - 1}）")
        out.append(i)
    else:
        hits = [i for i, n in enumerate(names) if n == tok] or \
               [i for i, n in enumerate(names) if n.startswith(tok)]
        if not hits:
            sys.exit(f"未知 case: {tok}")
        if len(hits) > 1:
            sys.exit(f"case 名不唯一: {tok} -> {[names[i] for i in hits][:5]}")
        out.append(hits[0])
print(" ".join(map(str, sorted(set(out)))))
PY
)" || exit 2

# 连续索引合并成 start:limit 段（bench.py 只支持区间，不支持任意索引集）
RUNS="$(python3 -c "
import sys
runs = []
for i in map(int, sys.argv[1].split()):
    if runs and i == runs[-1][0] + runs[-1][1]:
        runs[-1][1] += 1
    else:
        runs.append([i, 1])
print(' '.join(f'{s}:{n}' for s, n in runs))
" "$CASES")"

N_CASES="$(echo $CASES | wc -w)"
N_TREES="$(echo $TREES | wc -w)"
# 全量（0..105 一整段）时 bench.py 不带参数
bench_args() {  # $1=start $2=limit
  if [ "$1" = 0 ] && [ "$2" = "$N_CASES" ] && [ "$N_CASES" = 106 ]; then echo ""; else echo "--start $1 --limit $2"; fi
}

echo "=================================================================="
echo " KDA 三形态测试   卡=$CARD   形态=$DTYPE   性能采集=$WITH_PERF"
echo " case: $CASE_SPEC  ->  $N_CASES 条（索引: $(echo $CASES | cut -c1-60)$([ "$N_CASES" -gt 12 ] && echo ...)）"
echo " 段:   $RUNS"
echo " python: $(command -v python3)"
echo " 开始: $(date '+%F %T')"
echo "=================================================================="

for t in $TREES; do
  echo
  echo "########## $t ##########"

  # ── ① 全链正确性：每段一次 bench.py（一次跑完 6 个算子）──
  for r in $RUNS; do
    s="${r%:*}"; n="${r#*:}"; a="$(bench_args "$s" "$n")"
    (cd "$ROOT/$t" && source ./env.sh "$CARD" >/dev/null 2>&1 && cd bench && \
     echo "[①] 全链正确性 $a" && \
     python3 bench.py $a 2>&1 | tail -2)
  done

  # ── ② 全链性能（msprof，按同样分段）──
  if [ "$WITH_PERF" = 1 ]; then
    for r in $RUNS; do
      s="${r%:*}"; n="${r#*:}"; a="$(bench_args "$s" "$n")"
      pd="/tmp/prof_chain_$t"
      (cd "$ROOT/$t" && source ./env.sh "$CARD" >/dev/null 2>&1 && cd bench && \
       echo "[②] 全链性能（msprof）$a" && \
       rm -rf "$pd" && \
       VISIBLE_DEVICES="$CARD" bash run_cpu.sh --msprof "$pd" $a \
          --repeats 5 --warmup 2 >/dev/null 2>&1 && \
       python3 per_case_profile.py --latest-dir "$pd" --mean 2>&1 | tail -12 && \
       rm -rf "$pd")
    done
  fi

  # ── ③ 单算子正确性 ＋ ④ 单算子隔离性能（固定目标 case，与 --case 无关）──
  for k in turbo_gate_chunk_cumsum turbo_token_parallel turbo_inter_solve \
           turbo_recompute_w_u turbo_delta_rule_h turbo_gla_output; do
    out="$(cd "$ROOT/$t" && source ./env.sh "$CARD" >/dev/null 2>&1 && cd "$k" && \
           python3 test.py --target 2>&1 | grep -E "max_diff|=>" | tr '\n' ' ')"
    echo "[③] $k: ${out:-（无输出，检查上方 stderr）}"

    if [ "$WITH_PERF" = 1 ]; then
      pd="/tmp/prof_${t}_$k"
      out="$(cd "$ROOT/$t" && source ./env.sh "$CARD" >/dev/null 2>&1 && cd "$k" && \
             rm -rf "$pd"; mkdir -p "$pd" && \
             msprof --export=on --output="$pd" \
                 --application="python3 test.py --perf --repeats 7 --warmup 3" >/dev/null 2>&1 && \
             echo "  [④] $k: $(python3 test.py --report "$pd" 2>&1 | grep -E 'calls=' | tr '\n' ' ')" ; \
             rm -rf "$pd")"
      echo "$out"
    fi
  done
done

echo
echo "=================================================================="
echo " 完成: $(date '+%F %T')"
if [ "$DTYPE" = all ] && [ "$N_CASES" = 106 ]; then
  echo " 预期：三棵树均 PASS 627/636（失败 9 行 = 4 行 grid 超限 + 5 行 K=32 不支持）"
fi
echo "=================================================================="
