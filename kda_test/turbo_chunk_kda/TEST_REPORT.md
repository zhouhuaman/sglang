# KDA turbo 系列算子 · 交付测试报告与复现指南

## 1. 交付结构与命名

```
turbo_chunk_kda/
├── env.sh                              # 环境准备（source 一次，见 §3）
├── TEST_REPORT.md                      # 本文档
├── turbo_gate_chunk_cumsum/            # K1 门控激活 + chunk 内前缀和（→ log2 空间 gk）
│   ├── turbo_gate_chunk_cumsum_kernel.py   # torch 参考 + triton 实现（唯一代码）
│   └── test.py                         # 单算子：正确性 + 性能复现
├── turbo_token_parallel/               # K2 16-token 子窗口打分（Aqk / Akk）
├── turbo_inter_solve/                  # K3 块间耦合 + 整块下三角逆（Akk_inv）
├── turbo_recompute_w_u/                # K4 解耦表示 w / u + key 对齐 kg
├── turbo_delta_rule_h/                 # K5 跨 chunk 状态递推（h / v_new）
├── turbo_gla_output/                   # K6 输出合成 o = q·exp2(g)·h + Aqk·v_new
└── bench/                              # 6 算子串成全链的统一 bench（§5）
    ├── bench.py                        #   模式 A 全链正确性；模式 B msprof 分段采集
    ├── per_case_profile.py             #   msprof 产物 → results.csv
    ├── analyze_results.py              #   结果/加速比汇总
    ├── gen_cases.py / cases_meta.json  #   用例表（106 个，末位即目标 case）
    ├── run_cpu.sh                      #   模式 B 一键 msprof 采集
    ├── time_kernels.py                 #   迭代用 wall-clock 计时 + 精度
    ├── run_all_msprof_local.sh         #   全量 106 case msprof（分批）
    └── README.md                       # bench 细节
```

| 算子 | 目录 / 模块 | triton 入口 | torch 参考 | 上游链路角色 |
|---|---|---|---|---|
| K1 | `turbo_gate_chunk_cumsum/` `turbo_gate_chunk_cumsum_kernel.py` | `turbo_gate_chunk_cumsum_triton` | `turbo_gate_chunk_cumsum_torch` / `_ref` | 输出 `gk`（log2 空间），供 K2..K6 |
| K2 | `turbo_token_parallel/` `turbo_token_parallel_kernel.py` | `turbo_token_parallel_triton` | `turbo_token_parallel_torch` / `_ref` | 16×16 对角块 `Aqk`(留对角)/`Akk`(去对角) |
| K3 | `turbo_inter_solve/` `turbo_inter_solve_kernel.py` | `turbo_inter_solve_triton` | `turbo_inter_solve_torch` / `_ref` | 整块下三角逆 `Akk_inv` + 跨子块 `Aqk` |
| K4 | `turbo_recompute_w_u/` `turbo_recompute_w_u_kernel.py` | `turbo_recompute_w_u_triton` | `turbo_recompute_w_u_torch` / `_ref` | 解耦 `w`/`u`/`kg` |
| K5 | `turbo_delta_rule_h/` `turbo_delta_rule_h_kernel.py` | `turbo_delta_rule_h_triton` | `turbo_delta_rule_h_torch` / `_ref` | 跨 chunk 状态递推 `h`/`v_new`（唯一串行，K=V 约束） |
| K6 | `turbo_gla_output/` `turbo_gla_output_kernel.py` | `turbo_gla_output_triton` | `turbo_gla_output_torch` / `_ref` | 输出合成 `o` |

> 每个算子目录只有 **kernel 模块 + test.py** 两个文件，不 import 任何 sglang / 框架
> 代码，可独立拷走运行。

## 2. 环境与版本（验收机器 A5-29，2026-09-07 / 09-08 复测同口径）

| 项 | 值 |
|---|---|
| 主机 | `A5-29`（Linux 5.4.0-125-generic x86_64） |
| NPU | 昇腾 **Ascend 950PR** ×8 物理卡；验收测试用卡 3，复测用卡见 §7 |
| 驱动 | 25.7.rc1.b999（ascendhal 7.35.23） |
| CANN | **9.1.0**（`ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0`） |
| Python / torch / torch_npu | 3.11.10 / 2.10.0+cpu + npu 后端 / 2.10.0.post4 |
| triton-ascend | **3.2.1** |
| msprof | CANN 自带，`op_summary_*.csv` 的 `Task Duration(us)` 为权威耗时 |

## 3. 统一口径与准备

数据流（BT=64 chunk、BC=16 sub-chunk）：

```
g       = K1(x, A_log, dt_bias)                       # 门控激活 + chunk 前缀和
Aqk_d, Akk  = K2(q, k, g, beta, scale)                # 块内（对角线）得分
Aqk_nd, Akk_inv = K3(q, k, g, beta, Akkd=Akk, scale)  # 块间得分 + 下三角逆
w, u, kg = K4(k, v, beta, A=Akk_inv, gk=g)            # 解耦表示
h, v_new = K5(kg, w, u, gk=g, initial_state, idx)     # 跨 chunk 状态递推
o        = K6(q, v_new, g, Aqk=Aqk_d+Aqk_nd, h=h)     # 输出合成
```

- **目标 case（统一验收口径）**：`D_KV128_H96_T16384` → `B=1, T=16384, H=96, K=V=128`，
  fp32，NT=256。
- **正确性**：triton vs 同目录 torch 参考，各输出逐元素 `max_diff < 1e-2` PASS。
- **性能**：一律 `msprof` op_summary `Task Duration(us)` **每调用均值**；自报 wall-clock
  仅作快照。对比须同卡、同会话、同 repeats/warmup。
- **契约**：`chunk_size=64`（K2/K3 另 `sub_chunk_size=16`）不可改；fp32 主路径。

```bash
# ① 切到 3.2.1 环境（必须，见 §2 警示）
export PATH=/data/anaconda3/envs/autotriton/bin:$PATH     # 或: conda activate autotriton
python3 -c "import triton; print(triton.__version__)"      # 3.2.0 = triton-ascend 3.2.1 正常
# ② 进目录、选卡、准备环境
cd <turbo_chunk_kda 所在目录>
npu-smi info            # 挑空闲卡，避免与他人撞卡（并发报 ERR00100/Resource_Busy）
source env.sh           # CANN + torch/torch_npu 动态库 + 后端开关
source env.sh 4         # 推荐：直接固定到物理卡 4（npu-smi 确认空闲后）
```

## 4. 单算子测试

六个算子互相独立：`test.py` 自造输入、与同文件 torch 参考比对。**正确性秒级出结果，
性能复现（msprof）约 1-2 分钟/算子**。

### 4.1 逐算子命令（以 K1 为例；换目录名即测其它算子）

```bash
cd turbo_chunk_kda/turbo_gate_chunk_cumsum
source ../env.sh 4                      # 固定到空闲卡（按 npu-smi 结果自选）
python3 test.py                         # ① 正确性：输出 PASS + max_diff（门槛 <1e-2）
# ② 性能复现（msprof，官方唯一口径；测的就是目标 case）：
msprof --output=./prof_k1 --application="python3 test.py --perf --repeats 7 --warmup 3" \
    && python3 test.py --report ./prof_k1      # 解析 op_summary → Task Duration 每调用均值(ms)
python3 test.py --selftest              # ③ 选看：多 shape 边界自检（非验收项）
```

### 4.2 一键冒烟 / 逐算子基线

```bash
# 6 个算子全部正确性冒烟（每算子秒级）
cd turbo_chunk_kda && for d in turbo_*/; do
  (cd "$d" && echo "== $d" && python3 test.py | tail -2); done
# 逐算子 msprof 基线（每个约 1-2 分钟；产出 prof_kX/ 目录，比对后自行删除）
for d in turbo_gate_chunk_cumsum turbo_token_parallel turbo_inter_solve \
         turbo_recompute_w_u turbo_delta_rule_h turbo_gla_output; do
  (cd "$d" && source ../env.sh 4 && \
   msprof --output=./prof_$d --application="python3 test.py --perf --repeats 7 --warmup 3" \
     && python3 test.py --report ./prof_$d | tail -3); done
```

## 5. pipeline / bench 测试（6 算子全链）

`bench/` 用固定种子即时生成张量（不入库），逐 case 先以 torch 元算子链算出各算子输入，
再让 K_torch 与 K_triton 在完全相同输入下逐算子比对（`<1e-2` PASS），并把 6 算子
串成一个可 msprof 分段的调用序列（marker 分段）。

### 5.1 模式 A：全链正确性

```bash
cd turbo_chunk_kda/bench
source ../env.sh 4                       # 固定空闲卡
python3 bench.py --start 105 --limit 1           # ① 目标 case 全链正确性（106 条中末位）
python3 bench.py --limit 3                       # ② 冒烟：前 3 个 case
python3 bench.py                                 # ③ 全量 106 case（约 10-20 分钟）
# → correctness.csv：每 (case, K1..K6) 一行 status=PASS/FAIL + max_diff
```

### 5.2 模式 B：msprof 分段计时（性能唯一口径）

```bash
cd turbo_chunk_kda/bench
bash run_cpu.sh --msprof ./prof_target --start 105 --limit 1 \
     --repeats 5 --warmup 2                       # ① 目标 case msprof 采集
python3 per_case_profile.py --latest-dir ./prof_target --mean    # ② → results.csv
python3 analyze_results.py --pivot-case D_KV128_H96_T16384       # ③ 6 算子分段 ms 汇总
```

- `run_cpu.sh` 默认屏蔽物理卡 0（该卡曾硬故障）；覆盖：
  `VISIBLE_DEVICES=4,5 bash run_cpu.sh ...` 或 `DEVICE=npu:1 bash run_cpu.sh ...`。
- 迭代期墙钟验证：`python3 time_kernels.py`（`--kernel K3` 只测单算子）。
- 全量 106 case msprof 分批：`bash run_all_msprof_local.sh`。细节见 `bench/README.md`。

## 6. 验收记录（2026-09-07，triton-ascend 3.2.1，目标 case）

### 6.1 精度（6/6 PASS，另含 7 个边界 shape 全 PASS）

| Kernel | 参考对比 | 实测 max_diff | 门槛 | 结果 |
|---|---|---|---|---|
| K1 gate_chunk_cumsum | torch | 1.37e-04 | <1e-2 | ✅ PASS |
| K2 token_parallel（hm2） | torch（Aqk/Akk） | 2.98e-07 | <1e-2 | ✅ PASS |
| K3 inter_solve（NP=3） | torch（Aqk/Akk_inv） | 4.77e-07 | <1e-2 | ✅ PASS |
| K4 recompute_w_u | torch（w/u/kg） | 2.98e-04 | <1e-2 | ✅ PASS |
| K5 delta_rule_h | torch（h/v_new/终态） | 8.94e-07 | <1e-2 | ✅ PASS |
| K6 gla_output | torch（o） | 1.19e-07 | <1e-2 | ✅ PASS |

### 6.2 性能（msprof 每调用均值，ms）

| Kernel | ms/调用 | 8/31 design 对照 | 差异 |
|---|---|---|---|
| K1 | **1.373** | 1.380 | +0.5% |
| K2 | **5.548** | 5.621 | +1.3% |
| K3 | **10.696** | 10.693 | −0.03% |
| K4 | **3.623** | 3.609 | −0.4% |
| K5 | **11.431** | 11.425 | −0.05% |
| K6 | **3.912** | 3.914 | +0.05% |
| **合计（6-kernel 链）** | **36.58 ms** | 36.64 ms | +0.15% |

> 差异全部 <1.5%，确认环境还原与测量可复现。同 case 下 torch_npu 元算子链合计
> ≈0.91 s（K5 torch 单算子 ≈0.69 s），triton 6-kernel 相对 torch 拼接加速 **≈25×**
> （2026-09-08 bench 模式 B 同测口径，见 §7）。

### 6.3 与手写 Ascend C 融合算子（vllm-ascend `ChunkKdaFwd`）对比

| 实现 | 位宽 | 每调用 | 相对 |
|---|---|---|---|
| **Triton 6-kernel 分拆（本交付）** | fp32 | **36.58 ms** | 1.00× |
| Ascend C 融合（bf16） | bf16 | 37.22 ms | 1.02× |
| Ascend C 融合（fp16） | fp16 | 43.67 ms | 1.19× |

fp32 精度的分拆版与 bf16 融合算子持平、比 fp16 快 ~19%；两者均非 Cube 受限（带宽/
标量受限同源）。测量文档：`bench/chunk_kda_fwd_fused_bench.md`。

## 7. 实测记录（turbo 交付复测）

> 复测 2026-09-08，A5-29 物理卡 5（`ASCEND_RT_VISIBLE_DEVICES=5`），conda 环境
> `autotriton`（triton-ascend 3.2.1 + CANN 9.1.0），命令即 §4/§5 原文。改任一 kernel
> 后按同样命令重测并回填（注明日期/卡号/环境）。

### 7.1 单算子（对照 §6.1/§6.2）

| 算子 | test.py max_diff | msprof 隔离 ms/调用 | 官方基线 ms | 偏差 |
|---|---|---|---|---|
| K1 turbo_gate_chunk_cumsum | 1.373e-04 | 1.374 | 1.37 | +0.3% |
| K2 turbo_token_parallel | 2.980e-07 | 5.567 | 5.55 | +0.3% |
| K3 turbo_inter_solve | 4.768e-07 | 10.693 | 10.70 | −0.1% |
| K4 turbo_recompute_w_u | 2.720e-04 | 3.614 | 3.62 | −0.2% |
| K5 turbo_delta_rule_h | 8.941e-07 | 11.428 | 11.43 | ≈0% |
| K6 turbo_gla_output | 1.192e-07 | 3.911 | 3.91 | ≈0% |
| **合计** | — | **36.587** | 36.58 | ≈0% |

6/6 PASS；K2/K3/K5/K6 精度与官方验收逐位一致，K1/K4 的量级（1e-4）与官方一致
（官方 1.37e-04 / 2.98e-04，随随机种子波动）。

### 7.2 pipeline（bench）

- **模式 A 目标 case**：6/6 PASS（max_diff：K1 1.1e-05、K2 1.5e-07、K3 1.5e-07、
  K4 1.3e-04、K5 1.5e-08、K6 3.3e-09）；冒烟 3 case 18/18 PASS。
- **模式 B 目标 case**（msprof 分段，每调用均值 us）：

| | K1 | K2 | K3 | K4 | K5 | K6 | 合计 |
|---|---|---|---|---|---|---|---|
| triton us | 1375.96 | 5599.91 | 10691.62 | 3632.41 | 11429.63 | 3912.51 | **36642.05**（36.64 ms） |
| torch us | 9844.70 | 132160.59 | 41260.85 | 19470.08 | 693971.06 | 14176.46 | 910883.73 |
| speedup | 7.16× | 23.60× | 3.86× | 5.36× | 60.72× | 3.62× | ≈24.9× |

> 模式 B 与单算子隔离的差异 <0.6%（K2 +0.6% 最大，来自链式 marker 与进程内排序），
> 官方基线对照一致。
