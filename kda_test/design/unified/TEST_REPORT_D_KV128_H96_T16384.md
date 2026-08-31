# KDA 6 算子目标 case 测试报告

**case**: `D_KV128_H96_T16384`（B=1, T=16384, H=96, K=V=128）
**日期**: 2026-08-31
**状态**: ✅ K1–K6 全部通过（修复 K2/K5 编译与精度问题后全量复测）

---

## 1. 测试环境

| 组件 | 版本 |
|---|---|
| 硬件 | Ascend950PR（A5, ascend950pr_9579，可见 6 卡，跑在 npu:0） |
| CANN | 9.1.0 |
| triton-ascend | 3.2.1 |
| triton | 3.2.0 |
| torch / torch_npu | 2.10.0+cpu / 2.10.0.post2 |
| 容器 | triton-ascend-env-zhm（hostname A5-29） |

> 注：ANALYSIS.md §4.2「官方基线」标注硬件为 910B2，为旧文档记录；本机实测
> 为 Ascend950PR（`npu-smi` 确认），本次测试与 Ascend C 融合算子测试
> （§6）在同一台机器、同一 CANN 版本下完成，横向对比口径一致。

## 2. 测试方法

- **正确性**：`bench.py` 模式 A，K1–K6 各自用完全相同的输入（torch 链中间量）跑
  K_torch 与 K_triton，对比 `max_diff`，阈值 `< 1e-2`。
- **性能**：`run_cpu.sh --msprof ./prof_target_d3 --start 105 --limit 1
  --repeats 5 --warmup 2`，口径为 msprof `op_summary.csv` 的
  `Task Duration(us)` 每次调用均值（warmup=2 不计入，取后 5 次 repeat 均值）。
- 对比基线：ANALYSIS.md §4.2 官方基线（同机同态，CANN 9.0 / torch_npu 2.7.1）。

## 3. 正确性结果

| Kernel | max_diff | 判定 | 官方基线 max_diff |
|---|---|---|---|
| K1 gate_chunk_cumsum | 1.14e-05 | OK | 1.14e-05 |
| K2 token_parallel | **1.49e-07** | OK | 8.94e-08 |
| K3 inter_solve | 1.49e-07 | OK | 7.45e-08 |
| K4 recompute_w_u | 1.30e-04 | OK | 0.0 |
| K5 delta_rule_h | **1.49e-08** | OK | 0.0 |
| K6 gla_output | 9.33e-03 | OK（临界） | 3.73e-09 |

**6/6 PASS**。全部 `max_diff < 1e-2`。

## 4. 性能结果（msprof，每次调用均值 us）

| Kernel | torch_us | triton_us | speedup | 官方基线 triton_us | 偏差 |
|---|---|---|---|---|---|
| K1 | 9844.8 | **1379.96** | 7.13x | 1880.1 | **−26.6%** |
| K2 | 132299.5 | **5620.98** | 23.54x | 9416.3 | **−40.3%** |
| K3 | 41233.1 | **10692.67** | 3.86x | 14366.9 | **−25.6%** |
| K4 | 19477.9 | **3609.17** | 5.40x | 4514.1 | **−20.0%** |
| K5 | 696780.8 | **11425.37** | 60.99x | 9150.8 | **+24.9%** |
| K6 | 14191.0 | **3914.44** | 3.63x | 7385.5 | **−47.0%** |

**总 triton ≈ 36.64ms**（6 kernel 全量）。K1/K2/K3/K4/K6 均快于官方基线
（−20% ~ −47%）；**K5 慢于基线 +25%**（见 §6 说明）。

### 相对时间占比（triton_us）

| Kernel | tri% | 说明 |
|---|---|---|
| delta_rule_h (K5) | 31.2% | 序列递推串行关键路径 |
| inter_solve (K3) | 29.2% | 6-dot 逆链 |
| token_parallel (K2) | 15.3% | — |
| gla_output (K6) | 10.7% | fp32 带宽受限 |
| recompute_w_u (K4) | 9.8% | 向量受限 |
| gate_chunk_cumsum (K1) | 3.8% | 标量/向量均衡 |

## 5. 本轮修复内容（相对 8/25 基线）

### 5.1 K2 token_parallel —— 精度错误 + 慢 3×（已修复）

- **根因**：hm3 路径 kernel 内 `tl.gather(Akk_full, col_idx, axis=1)` 的源为
  `tl.dot` 输出时，triton-ascend 3.2.1 codegen 结果错误（最小复现实验：
  load 源正确 0.0 / dot 源错误）。Akk max_diff 0.32、耗时 29.1ms。
- **修复**：driver 切回 hm2 路径（满宽写 scratch + driver `torch.gather` 收拢，
  数学一致），hm3 保留标注 DEPRECATED。
- **效果**：max_diff 0.32 → **1.49e-7**；耗时 29.09ms → **5.62ms**（比官方基线快 40%）。

### 5.2 K5 delta_rule_h —— 编译失败（已修复）

- **根因 1（目标 case，K=128）**：`b_h += tl.trans(tl.dot(k, b_v))` 的
  dot 输出转置+累加链触发 CANN 9.1 hivm-plan-memory
  `"Unsupported op for finding the root alloc"`，报错
  `ub overflow, requires 2097152 bits`（256KB 为固定误报值，与 BV/NS 无关）。
- **根因 2（K=64 路径）**：`_store_h_full` flat 1D store 的
  `+zeros`+`reshape` workaround 触发 `expand_shape collapsed dim size 2048
  must equal 4096`。
- **修复**：外积改写为 `b_h += tl.dot(tl.trans(b_v), b_k2)`（行主 `[BT,K]`
  加载 + 输入侧转置，数学等价）；`_store_h_full` 统一为 2D 手动指针 store。
- **效果**：编译通过，max_diff **1.49e-8**（恢复基线精度）；K=64 小 case 亦修复。

### 5.3 工具修复

- `per_case_profile.py`：中间 kernel 失败导致 marker 分段错位（K5 段被标成 K6）
- `analyze_results.py`：kernel id（K1..K6）与全名匹配不一致导致全部显示 unsupported
- 新增 `k2_debug.py`：K2 独立精度定位脚本；`tmp_k2_gather_test.py`：
  tl.gather 行为最小复现

## 6. Ascend C 融合算子对照（vllm-ascend ChunkKdaFwd）

### 6.1 测试对象与方法

与 triton 6-kernel 分拆实现对应的，是 vllm-ascend 仓库中的**融合 Ascend C 算子**
`chunk_kda_fwd`（Gate/Prepare/PostWu/FwdH/Finalize 在一个物理算子内完成，L2 负责
BSND→BNSD 布局转换）。完整测试见 `kda_test/chunk_kda_fwd_fused_bench.md`，
此处为摘要（同机 A5、同 CANN 9.1.0，2026-08-31 测试）：

| 项 | 值 |
|---|---|
| 硬件/软件 | Ascend950PR ×6 / CANN 9.1.0 / torch_npu 2.10.0.post2 / vllm_ascend 0.19.1rc2 |
| 计时 | msopprof per-call 权威 + torch.npu Event 快速对照 |
| 输入 | q/k/v = randn×0.05, g = −rand×0.05, beta = sigmoid(randn) |
| 正确性 | T=1024 CPU 参考对照，max_abs = 1.9e-6 通过 |

### 6.2 目标 case 结果（B=1, T=16384, H=96, K=V=128）

| 配置 | msopprof per-call | 吞吐 | 相对 |
|---|---|---|---|
| fp16, chunk=64（目标） | **43.67 ms** | ~375 K tokens/s | 1.00× |
| bf16, chunk=64 | 37.22 ms | ~440 K tokens/s | 0.85× |
| fp16, chunk=128 | 51.24 ms | ~320 K tokens/s | 1.17× |

单次调用物理 launch 分解（211 次统计，极稳定）：主 kernel `KdaChunkForward` #0
**32.88 ms（占 75%）** + 同名 #1/#2/#3 共 10.78 ms + `TransposeAiCore` 0.017 ms，
per-call 合计 43.67 ms。

### 6.3 triton 6-kernel vs Ascend C 融合算子

| 指标 | triton 6-kernel（本文 §3/§4） | Ascend C 融合（chunk=64） |
|---|---|---|
| 计算精度 | fp32 | fp16 / bf16 |
| **总耗时** | **36.64 ms**（6 kernel 求和） | **43.67 ms**（fp16）/ **37.22 ms**（bf16） |
| 正确性 | 6/6 OK，max_diff < 1e-2（vs torch_npu 链） | max_abs 1.9e-6（vs CPU 参考） |
| 结构 | 6 独立 kernel + torch 中间拼接 | 单算子，L2 内 4+1 次物理 launch |
| 精度基准 | fp32 全链 | fp16/bf16 全链 |

观察：

1. **同数量级，triton 分拆版略快**：fp32 36.6ms < fp16 融合 43.7ms。triton 版
   未做任何低精度优化（全 fp32），融合算子用 fp16 反而更慢，说明两者均
   **memory/流水线受限而非计算受限**（融合算子 6.5 节手算有效算力 ~7 TFLOPS，
   远低于 A5 理论值）。
2. **bf16 比 fp16 快 15%**（融合算子内实测）：带宽敏感特征，Kimi K3 实际部署
   （bf16）有利；triton 版暂无 bf16 对比。
3. **主 kernel 占 75%**：融合算子的 #0 kernel（32.9ms）与 triton 版耗时结构
   可对照——triton 版中 K5(11.4ms)+K3(10.7ms)+K2(5.6ms) 为主要耗时，融合
   算子的 FwdH/PostWu 阶段对应其中 chunk 间串行部分。
4. **chunk=64 优于 128**（融合算子实测 51.2ms vs 43.7ms）：O(chunk²) intra-chunk
   成本主导，与 triton 版 BT=64 的选择一致。

## 7. 遗留问题

1. **K5 性能 +25%**（11.43ms vs 官方 9.15ms）：CANN 9.1 下原写法无法编译，
   输入侧转置形态为实测最优可用方案（曾试 3 种替代：`trans(dot)`+where 打断
   21.8ms、状态转置形态 22.3ms、NS=2/3/4 与 NW=4/8 无差异）。
2. **K6 精度退化**：9.33e-3（官方 3.73e-9）——8/31 commit 的 HM=16 标量
   饱和削减的代价，仍在 1e-2 阈值内但余量仅 7%。A 组小 T case
   （T=1024/2048）K6 max_diff 达 1.34e-2~1.44e-2，**超阈值 FAIL**。
3. **K4 精度**：1.30e-4（基线 0.0，bitwise 一致），有所退化但远低于阈值，
   未定位原因。
4. **全量 106 case 未跑**：本次仅目标 case；全量正确性/性能采集可后续
   `bash run_all_msprof_local.sh` 分批执行（T≥65536 单 case 一批）。
5. **Ascend C 融合算子覆盖限制**：仅测 `use_gate_in_kernel=False`（raw gate）
   路径；`safe_gate`/`use_gate_in_kernel=True`/`initial_state`/变长序列未覆盖
   （见 chunk_kda_fwd_fused_bench.md §8）；FLOPs 为手算粗估。

## 8. 复现方法

```bash
# 正确性（目标 case）
python3 bench.py --start 105 --limit 1

# 性能采集（msprof）
bash run_cpu.sh --msprof ./prof_target_d3 --start 105 --limit 1 --repeats 5 --warmup 2
python3 per_case_profile.py --latest-dir ./prof_target_d3 --mean
python3 analyze_results.py --pivot-case D_KV128_H96_T16384

# K2 独立精度定位
python3 k2_debug.py

# Ascend C 融合算子（chunk_kda_fwd_fused_bench.md §7 摘要）
python3 prof_chunk_kda_fwd_fused.py --check                    # 正确性抽查
python3 prof_chunk_kda_fwd_fused.py                            # fp16 event 计时
msprof --application="python3 prof_chunk_kda_fwd_fused.py --iters 200 --warmup 10" \
       --output=/tmp/kda_prof
python3 prof_chunk_kda_fwd_fused.py --parse /tmp/kda_prof      # msopprof 解析
```

原始数据：triton 版 msprof trace 在 `prof_target_d3/`，解析结果 `results.csv`，
正确性 `correctness.csv`；Ascend C 版见 `kda_test/chunk_kda_fwd_fused_bench.md`。
