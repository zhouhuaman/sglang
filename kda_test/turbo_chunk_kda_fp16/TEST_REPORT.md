# KDA turbo 系列算子 · fp16 输入 测试报告与复现指南

## 1. 目录结构

```
turbo_chunk_kda_fp16/
├── env.sh                              # 环境准备（source 一次，见 §4）
├── TEST_REPORT.md                      # 本文档（fp16 形态版）
├── EXPERIMENT_FP16.md                  # 逐算子实验记录、归因实验、动态范围分析
├── turbo_gate_chunk_cumsum/            # K1  gate 激活 + chunk 前缀和（无 dot）
├── turbo_token_parallel/               # K2  16×16 子窗口打分（fp16 dot）
├── turbo_inter_solve/                  # K3  块间耦合 + 下三角逆（fp32 cube，见 §3）
├── turbo_recompute_w_u/                # K4  解耦表示 w/u/kg（fp16 dot）
├── turbo_delta_rule_h/                 # K5  状态递推 h/v_new（保持 fp32，见 §3）
├── turbo_gla_output/                   # K6  输出合成（fp16 dot）
└── bench/                              # 6 算子全链 bench（模式 A/B，用法同 fp32 版）
```

每个算子目录仍是 **kernel 模块 + test.py** 两个文件；入口函数名 `turbo_<op>_{triton,torch,ref}`
与 fp32 版一致，调用方传 fp32 或 fp16 均可（driver 内部按各算子形态转换，不改调用方张量；
K5 的 `initial_state` 就地更新语义保持）。

## 2. 目标与口径（与 fp32 版一致）

- 目标 case：`D_KV128_H96_T16384`（B=1, T=16384, H=96, K=V=128），NT=256。
- 正确性：triton vs 同目录 torch fp32 参考，逐元素 `max_diff < 1e-2` 判 PASS（**注意**：
  fp16 形态下个别算子贴近门槛，见 §5 精度余量说明）。
- 性能：msprof op_summary `Task Duration(us)` 每调用均值（warmup 3 + repeats 7）。
- 契约：`chunk_size=64` / `sub_chunk_size=16` 不可改。

## 3. 各算子 dtype 形态（最终采纳）

| 算子 | 输入张量 | 参数(A_log/dt_bias/scale) | vector 段 | tl.dot(cube) | 输出 | 理由 |
|---|---|---|---|---|---|---|
| K1 | x→**fp16** | **fp32** | fp32 | 无 dot | **fp32** | 见 §5-① |
| K2 | q/k/gk/beta→**fp16** | — | fp32 | **fp16**(+饱和 clamp) | **fp16** | fp16 dot + 带宽双赢 |
| K3 | q/k/g/beta→**fp16** | — | fp32 | **fp32**(工具链缺陷阻断) | **fp16** | 见 §5-③ |
| K4 | k/v/beta/A/gk→**fp16** | — | fp32 | **fp16** | **fp16** | 收益最大 |
| K5 | —（**整体保持 fp32 原版**） | — | fp32 | fp32 | fp32 | 见 §5-④ |
| K6 | q/v_new/g/Aqk/h→**fp16** | — | fp32 | **fp16** | **fp16** | fp16 dot + 带宽双赢 |

K2 的 fp16 饱和（±65504）只作用于溢出区，正常量级数值不受影响。

## 4. 环境与命令

环境与 fp32 版完全相同（**必须 conda `autotriton`，triton-ascend 3.2.1**）：

```bash
export PATH=/data/anaconda3/envs/autotriton/bin:$PATH   # 必须: 默认 python 是 3.2.2+dev
cd <turbo_chunk_kda_fp16 所在目录>
npu-smi info                                             # 挑空闲卡
source env.sh 5                                          # 固定物理卡 5（按实际改）
```

单算子（每算子目录独立自测，与 fp32 版同命令）：

```bash
cd turbo_gate_chunk_cumsum
source ../env.sh 5
python3 test.py                       # ① 正确性：PASS + max_diff（门槛 <1e-2）
msprof --output=./prof_k1 --application="python3 test.py --perf --repeats 7 --warmup 3" \
    && python3 test.py --report ./prof_k1     # ② msprof 每调用均值(ms)
```

全链 bench：

```bash
cd turbo_chunk_kda_fp16/bench
source ../env.sh 5
python3 bench.py --start 105 --limit 1         # 模式 A：目标 case 全链（注意 §5-② K2）
bash run_cpu.sh --msprof ./prof_target --start 105 --limit 1 --repeats 5 --warmup 2 \
    && python3 per_case_profile.py --latest-dir ./prof_target --mean \
    && python3 analyze_results.py --pivot-case D_KV128_H96_T16384   # 模式 B
```

## 5. 实测结果（2026-09-09，卡 5，triton-ascend 3.2.1）

### 5.1 单算子（正确性 + msprof 隔离）

| 算子 | max_diff | 门槛 | fp16 版 ms | fp32 基线 ms | 变化 | 结论 |
|---|---|---|---|---|---|---|
| K1 gate_chunk_cumsum | 9.58e-3 | <1e-2 | **1.195** | 1.374 | **−13.0%** |
| K2 token_parallel | 5.81e-4 | <1e-2 | **4.122** | 5.567 | **−26.0%** |
| K3 inter_solve | 4.98e-4 | <1e-2 | 10.790 | 10.693 | +0.9% |
| K4 recompute_w_u | 3.67e-3 | <1e-2 | **1.998** | 3.614 | **−44.7%** |
| K5 delta_rule_h | 8.94e-7 | <1e-2 | 11.428 | 11.428 | 0%（fp32 原版） |
| K6 gla_output | 2.05e-4 | <1e-2 | **2.303** | 3.911 | **−41.1%** |
| **合计** | 6/6 PASS | — | **≈31.84** | 36.587 | **−13.0%** |

### 5.2 全链 bench（模式 A 目标 case）

| K1 | K2 | K3 | K4 | K5 | K6 | 结果 |
|---|---|---|---|---|---|---|
| 3.98e-3 OK | 3.81e-2 OK | 6.86e-4 OK | 2.83e-3 OK | 1.49e-8 OK | 7.94e-6 OK | **6/6** |
