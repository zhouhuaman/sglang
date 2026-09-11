# KDA turbo 系列算子 · bf16 输入 测试报告与复现指南

## 1. 目录结构

```
turbo_chunk_kda_bf16/
├── env.sh                              # 环境准备（source 一次，见 §4）
├── TEST_REPORT.md                      # 本文档（bf16 形态版）
├── EXPERIMENT_BF16.md                  # 转制/扫描/失败算子处理记录
├── turbo_gate_chunk_cumsum/            # K1  保持 fp32 原版（见 §3）
├── turbo_token_parallel/               # K2  bf16 全链（含 bf16 cube）
├── turbo_inter_solve/                  # K3  输入/输出 bf16，cube fp32（见 §3/§6-③）
├── turbo_recompute_w_u/                # K4  保持 fp32 原版（见 §3）
├── turbo_delta_rule_h/                 # K5  保持 fp32 原版（fp16 版已证负收益）
├── turbo_gla_output/                   # K6  bf16 全链（含 bf16 cube）
└── bench/                              # 6 算子全链 bench（模式 A/B）
```

## 2. 目标与口径

- 目标 case：`D_KV128_H96_T16384`（B=1, T=16384, H=96, K=V=128）。
- 正确性：triton vs 同目录 torch fp32 参考，逐元素 `max_diff < 1e-2` PASS。
- 性能：msprof `Task Duration(us)` 每调用均值（warmup 3 + repeats 7）。
- 契约：`chunk_size=64` / `sub_chunk_size=16` 不可改。

## 3. 各算子 dtype 形态（最终采纳）

| 算子 | 输入 | 参数 | vector 段 | tl.dot(cube) | 输出 |
|---|---|---|---|---|---|---|
| K1 | **fp32（原版）** | fp32 | fp32 | 无 dot | fp32 |
| K2 | **bf16** | — | fp32 | **bf16**（无需饱和） | **bf16** |
| K3 | **bf16** | — | fp32 | **fp32**（B 形态） | **bf16** |
| K4 | **fp32（原版）** | — | fp32 | fp32 | fp32 |
| K5 | **fp32（原版）** | — | fp32 | fp32 | fp32 |
| K6 | **bf16** | — | fp32 | **bf16** | **bf16** |

## 4. 环境与命令

环境同 fp16/fp32 版（**必须 conda `autotriton`，triton-ascend 3.2.1**）：

```bash
export PATH=/data/anaconda3/envs/autotriton/bin:$PATH   # 必须
cd <turbo_chunk_kda_bf16 所在目录>
npu-smi info && source env.sh 5                          # 挑空闲卡并固定
cd turbo_token_parallel && source ../env.sh 5
python3 test.py                       # ① 正确性（门槛 <1e-2）
msprof --output=./prof_k2 --application="python3 test.py --perf --repeats 7 --warmup 3" \
    && python3 test.py --report ./prof_k2     # ② msprof 每调用均值
cd ../bench && python3 bench.py --start 105 --limit 1   # ③ 全链正确性（模式 A）
```

## 5. 实测结果（2026-09-09，卡 5，triton-ascend 3.2.1）

### 5.1 单算子

| 算子 | 形态 | max_diff | bf16 版 ms | fp32 基线 ms | 变化 | 结论 |
|---|---|---|---|---|---|---|
| K1 | fp32 | 1.37e-4 | 1.374 | 1.374 | 0% | fp32 原版（bf16 输入 FAIL 见 §6-①） |
| K2 | bf16 | 4.06e-3 | **3.826** | 5.567 | **−31.3%** | ✅ |
| K3 | bf16 in/out + fp32 cube | 3.61e-3 | 10.792 | 10.693 | +0.9% | ⚠ bf16 dot 被工具链缺陷阻断 |
| K4 | fp32 | 2.72e-4 | 3.614 | 3.614 | 0% | fp32 原版（bf16 dot FAIL 见 §6-②） |
| K5 | fp32 | 8.94e-7 | 11.428 | 11.428 | 0% | fp32 原版 |
| K6 | bf16 | 1.78e-3 | **2.381** | 3.911 | **−39.1%** | ✅ |
| **合计** | 6/6 PASS | — | **≈33.42** | 36.587 | **−8.7%** | 按各算子采纳形态 |

### 5.2 全链 bench（模式 A 目标 case）—— **6/6 PASS**

| K1 | K2 | K3 | K4 | K5 | K6 |
|---|---|---|---|---|---|
| 1.14e-5 OK | 6.03e-3 OK | 5.83e-3 OK | 1.30e-4 OK | 1.49e-8 OK | 5.94e-5 OK |

> 对比：fp16 版全链 5/6（K2 因 gk<−16 饱和超门槛）；bf16 版范围同 fp32，**无饱和
> 需求、全链干净通过**——这是 bf16 相对 fp16 在本链路的最大优势。

