#!/usr/bin/env python3
"""聚合 msprof metric_summary（prof_K2/K3/K5/K6b），输出各 kernel 的瓶颈指标。"""
import glob
import sqlite3

for d in ("prof_K2", "prof_K3", "prof_K5", "prof_k6b"):
    ps = glob.glob(f"{d}/PROF_*/*/sqlite/metric_summary.db")
    if not ps:
        print(f"{d}: no db", flush=True)
        continue
    con = sqlite3.connect(ps[0])
    rows = con.cursor().execute("""select task_id, count(*) ncores,
       sum(aic_total_cycles) aic_cyc, max(aic_mac_ratio_extra) aic_mac,
       max(aic_scalar_ratio) aic_scalar, max(aic_mte2_ratio) aic_mte2,
       max(aic_fixpipe_ratio) aic_fix,
       sum(aiv_total_cycles) aiv_cyc, max(aiv_vec_ratio) aiv_vec,
       max(aiv_scalar_ratio) aiv_scalar, max(aiv_mte1_ratio) aiv_mte1,
       max(aiv_mte2_ratio) aiv_mte2, max(aiv_mte3_ratio) aiv_mte3
       from MetricSummary
       where aic_total_cycles>0 or aiv_total_cycles>0
       group by task_id
       order by (sum(aic_total_cycles)+sum(aiv_total_cycles)) desc
       limit 4""").fetchall()
    print(f"=== {d} ===", flush=True)
    for r in rows:
        tid, nc, ac, amac, ascl, amte2, afix, av, avec, avscl, am1, am2, am3 = r
        print(f"  task={tid:<5} cores={nc:>3} aic_cyc={ac:>10} mac={amac*100:5.1f}% "
              f"scalar={ascl*100:5.1f}% mte2={amte2*100:5.1f}% | "
              f"aiv_cyc={av:>10} vec={avec*100:5.1f}% scl={avscl*100:5.1f}% "
              f"mte1={am1*100:5.1f}% mte2={am2*100:5.1f}% mte3={am3*100:5.1f}%", flush=True)
