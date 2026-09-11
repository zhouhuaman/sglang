#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Run full 105-case msprof in adaptive batches, merging results.csv.
# - T ≤ 8192: batch of 6 cases
# - T = 16384-32768: batch of 3 cases
# - T ≥ 65536: batch of 1 case
set -e
# 但 per_case_profile.py 对 <2 marker 的段会 SystemExit(1)；用 || true 兜底

UNIFIED_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$UNIFIED_DIR"

REPEATS=3
WARMUP=1
TOTAL_CASES=105
# 已完成批次（从 prof_batch_*/op_summary 存在判断），断点续跑
RESUME=${RESUME:-0}

# Clean previous batch artifacts (only when not resuming)
if [ "$RESUME" -eq 1 ]; then
  echo "=== 续跑模式：保留已有 prof_batch_*/ 与 results_batch_*.csv ==="
else
  rm -f results_all.csv results_batch_*.csv profile_meta_batch_*.json
  rm -rf prof_batch_*
fi

batch=0
start=0
while [ $start -lt $TOTAL_CASES ]; do
  remaining=$((TOTAL_CASES - start))
  
  # Determine batch size based on the T value of the starting case
  # Read the case's T from cases_meta.json
  case_t=$(python3 -c "
import json
with open('cases_meta.json') as f:
    meta = json.load(f)
cases = list(meta.items())
if $start < len(cases):
    print(cases[$start][1]['T'])
else:
    print(0)
")
  
  if [ "$case_t" -ge 65536 ]; then
    batch_size=1
  elif [ "$case_t" -ge 16384 ]; then
    batch_size=3
  else
    batch_size=6
  fi
  
  if [ $remaining -lt $batch_size ]; then batch_size=$remaining; fi
  
  batch=$((batch + 1))
  prof_dir="prof_batch_${batch}"
  end=$((start + batch_size - 1))
  echo "=== Batch $batch: cases $start..$end (T=$case_t, batch_size=$batch_size) -> $prof_dir ==="

  # 若已存在 op_summary 且续跑模式，跳过 msprof 采集，只重解析
  op_sum=$(find "./$prof_dir" -name "op_summary_*.csv" 2>/dev/null | head -1)
  if [ "$RESUME" -eq 1 ] && [ -n "$op_sum" ]; then
    echo "  (已有 op_summary，跳过 msprof，直接解析)"
  else
    # Run msprof for this batch
    docker exec -w "$UNIFIED_DIR" triton-ascend-env-zhm bash run_cpu.sh \
      --msprof "./$prof_dir" --start "$start" --limit "$batch_size" \
      --repeats "$REPEATS" --warmup "$WARMUP" 2>&1 | tail -1
    # bench.py 把 profile_meta 写到固定路径，改名到 per-batch
    # (msprof 可能吞掉 bench.py 的写文件；fallback 重建在下面)
    if [ -f profile_meta.json ]; then
      mv -f profile_meta.json "profile_meta_batch_${batch}.json"
    fi
  fi

  # 确保 profile_meta_batch_N.json 存在（续跑时可能缺失）
  if [ ! -f "profile_meta_batch_${batch}.json" ]; then
    echo "  [!] profile_meta_batch_${batch}.json 缺失，从 _kernel_supports 重建"
    python3 -c "
import json, importlib.util
spec=importlib.util.spec_from_file_location('bench','bench.py')
b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
meta=json.load(open('cases_meta.json')); cases=list(meta.items())
seg=cases[$start:$start+$batch_size]
pm=[]
for cid,m in seg:
    B,T,H,K,V=m['B'],m['T'],m['H'],m['K'],m['V']
    for km in b.KERNEL_META:
        kid=km['id']; sup,reason=b._kernel_supports(kid,B,T,H,K,V)
        pm.append({'case_id':cid,'kernel':kid,'triton_op':km['triton_op'],'repeats':0 if not sup else $REPEATS,'skipped':not sup,'reason':reason})
open('profile_meta_batch_${batch}.json','w').write(json.dumps(pm,ensure_ascii=False,indent=2))
"
  fi

  # Parse this batch's op_summary
  # (用 || true 兜底：若 marker 不足 2 个，per_case_profile.py 会 SystemExit)
  python3 per_case_profile.py --latest-dir "./$prof_dir" \
    --profile-meta "profile_meta_batch_${batch}.json" \
    --out "results_batch_${batch}.csv" 2>&1 | tail -2 || true

  # 若 results_batch_N.csv 缺失（parse 失败），创建只含表头的空文件
  if [ ! -f "results_batch_${batch}.csv" ]; then
    echo "case_id,B,T,H,K,V,kernel,torch_us,triton_us,speedup,max_diff,status" > "results_batch_${batch}.csv"
    # 补 skipped 段
    python3 -c "
import json,csv
pm=json.load(open('profile_meta_batch_${batch}.json'))
with open('results_batch_${batch}.csv','a',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['case_id','B','T','H','K','V','kernel','torch_us','triton_us','speedup','max_diff','status'])
    for p in pm:
        if p.get('skipped'):
            w.writerow({'case_id':p['case_id'],'B':'','T':'','H':'','K':'','V':'','kernel':p['kernel'],'torch_us':'-','triton_us':'-','speedup':'-','max_diff':'','status':p.get('reason','不支持')})
"
    echo "  [!] batch ${batch} parse 失败，创建空 results"
  fi
  
  # Append to combined results
  if [ $batch -eq 1 ]; then
    cat "results_batch_${batch}.csv" > results_all.csv
  else
    tail -n +2 "results_batch_${batch}.csv" >> results_all.csv
  fi
  
  rows=$(wc -l < results_all.csv)
  echo "Batch $batch done. Combined rows: $rows"
  
  start=$((start + batch_size))
done

# Final results.csv = combined
mv results_all.csv results.csv
echo ""
echo "=== All batches done. results.csv has $(wc -l < results.csv) rows ==="
python3 analyze_results.py 2>&1 | tail -50
