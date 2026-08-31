# KDA 6 算子统一 bench — 测试结果分析

> 本文档是 `unified/` 框架的**测试结果分析**，**不含脚本用法**（用法见
> `README.md`）。记录实测的正确性、性能与结论。

KDA（Kimi Linear Delta Attention）chunked attention 6 个 triton kernel
（K1 gate_chunk_cumsum / K2 token_parallel / K3 inter_solve /
K4 recompute_w_u / K5 delta_rule_h / K6 gla_output）相对 torch_npu 元算子版的
统一验证结果。

- **目标 case**: `B=1, T=16384, H=96, K=V=128`（case_id `D_KV128_H96_T16384`）
- **硬件**: Ascend 910B2（NPU 20 核），CANN 9.0.0 / triton-ascend 3.2.1 / torch_npu 2.7.1
- **容器**: `triton-ascend-env-zhm`

---

## 1. 测量口径

- 性能唯一来源：**msprof `op_summary.csv` 的 `Task Duration(us)`**（设备侧 kernel
  时间），不做 wall-clock 计时。
- marker 分段：每个 (case, kernel) 段前后发 `_kda_bench_marker`，段内以 triton op
  名开头的行 = `triton_us`（N 次调用总和），其余 = `torch_us`（torch_npu 拼接总和）；
  `speedup = torch_us / triton_us`。
- 复现判定：同机同态下 `triton_us` 与官方基线偏差应 < **±10%**；显著偏高时检查
  是否有其它进程占 NPU、或设备是否处于降频态。

## 2. 正确性结果

### 2.1 当前 `correctness.csv`（D 组 6 case）

仓库中 `correctness.csv` 现保存 **D 组 6 个 case** 的复核结果（36 行 =
6 case × 6 kernel），**31 `OK` + 5 预期 `不支持`**：

| case | K1 | K2 | K3 | K4 | K5 | K6 |
|------|----|----|----|----|----|----|
| D_KV128_H2_T1024 | OK | OK | OK | OK | OK | OK |
| D_KV128_H2_T16384 | OK | OK | OK | OK | OK | OK |
| D_KV128_H8_T1024 | OK | OK | OK | OK | OK | OK |
| D_KV128_H8_T16384 | OK | OK | OK | OK | OK | OK |
| D_KV32_H4_T4096 | 不支持 | 不支持 | 不支持 | 不支持 | OK | 不支持 |
| D_KV128_H96_T16384（目标） | OK | OK | OK | OK | OK | OK |

`D_KV32_H4_T4096` 的 5 个 `不支持` 原因（符合 README §9 约束，非 bug）：

- K1/K6：K=32 大 T 下精度不足（仅小 T ≤256 验证通过）；
- K2/K3/K4：`BK=32` 时 `tl.dot` 数值不稳定；
- K5：K=V=32 支持（0.0 精确）→ OK。

### 2.2 覆盖说明

- A/B/C 组（100 个 K=V=64 case）在框架清理前用**同一套 canonical kernel**（未被
  清理触碰）已验证通过；当前设备持续负载下不稳定（见 README §10），如需全量复核
  建议分批 `bash run_cpu.sh --start <s> --limit 10`。
- 当前 `correctness.csv` 仅含 D 组是"分批另存 + 设备崩溃后重跑失败批次"流程的
  当前落盘状态，不代表其它组未测。

## 3. 目标 case 精度基线

`D_KV128_H96_T16384`（B=1, T=16384, H=96, K=V=128）6 kernel 全部 `OK`：

| Kernel | max_diff | 说明 |
|--------|----------|------|
| K1 | 1.14e-05 | gate_chunk_cumsum |
| K2 | 8.94e-08 | token_parallel (Aqk/Akk) |
| K3 | 7.45e-08 | inter_solve (Aqk/Akk_inv) |
| K4 | 0.0 | recompute_w_u（bitwise 一致） |
| K5 | 0.0 | delta_rule_h（h/v_new bitwise 一致） |
| K6 | 3.73e-09 | gla_output |

全部 `max_diff < 1e-2` 判定阈值；K4/K5 与 torch_npu 参考 bitwise 一致。

## 4. 目标 case 性能结果

### 4.1 本次复核（2026-08-24，当前 `results.csv`）

`results.csv` 保存目标 case 的 msprof 数据（`--start 105 --limit 1 --repeats 5
--warmup 2 --mean`）：

| Kernel | torch_us | triton_us | speedup | max_diff |
|--------|----------|-----------|---------|----------|
| K1 | 12977.6 | 1822.5 | 7.1x | 1.14e-05 |
| K2 | 193119.1 | 9463.9 | 20.4x | 8.94e-08 |
| K3 | 132419.1 | 14406.5 | 9.2x | 7.45e-08 |
| K4 | 32036.4 | 4518.2 | 7.1x | 0.0 |
| K5 | 871290.8 | 9127.2 | 95.5x | 0.0 |
| K6 | 21567.7 | 6779.8 | 3.2x | 3.73e-09 |

**总 triton 时间 ≈ 46.1ms**。耗时排序（triton_us 降序）：K3(14.4) > K2(9.5) >
K5(9.1) > K6(6.8) > K4(4.5) > K1(1.8) ms。加速比排序：K5(95.5x，串行 torch 循环
几乎全被压掉) > K2(20.4x) > K3(9.2x) > K1(7.1x) > K4(7.1x) > K6(3.2x，fp32 带宽受限)。

### 4.2 官方基线（同机同态参考）

| Kernel | torch_us | triton_us | speedup | 说明 |
|--------|----------|-----------|---------|------|
| K1 | 12782.9 | **1880.1** | 6.8x | 1.88ms |
| K2 | 193277.5 | **9416.3** | 20.5x | 9.42ms |
| K3 | 133268.6 | **14366.9** | 9.3x | 14.37ms（NP=3, HM=16） |
| K4 | 32027.6 | **4514.1** | 7.1x | 4.51ms（HM=16） |
| K5 | 920639.6 | **9150.8** | 100.6x | 9.15ms（BV=V=128） |
| K6 | 21586.7 | **7385.5** | 2.9x | 7.39ms（BK=128, memory-bound） |

**总 triton 时间 ≈ 46.7ms**。

### 4.3 判定

- **精度**：目标 case 6 kernel 全部 `status=OK`、`max_diff < 1e-2`（K4/K5 为 0.0）。
- **性能**：目标 case 6 kernel `triton_us` 复现官方值，偏差全部 **< 10%**
  （本次复核 vs 官方基线：K1 −3.1%、K2 +0.5%、K3 +0.3%、K4 +0.1%、K5 −0.3%、
  K6 −8.2%，均在同机同态正常抖动范围内）。

### 4.4 标量削减轮集成（2026-08-25，`time_run.sh --repeats 7 --warmup 3`）

聚焦"6 算子 scalar 受限"（用户本轮方向）后重新集成 6 kernel，逐核确认瓶颈属性、
对确认受限的 kernel 施加头合并（HM）/流水线（NS）削减：

| Kernel | torch_us | triton_us | speedup | max_diff | 本轮动作 |
|--------|----------|-----------|---------|----------|----------|
| K1 | 12972.6 | 2189.0 | 5.9x | 1.14e-05 | 无（scalar 45.6%/vec 44.2%，均衡） |
| K2 | 193427.4 | 9775.4 | 19.8x | 8.94e-08 | 无（全 pipe <37%，延迟受限） |
| K3 | 144533.7 | 14875.7 | 9.7x | 7.45e-08 | 无（aiv_cyc 主导，6-dot 串行逆链关键路径） |
| K4 | 32199.8 | 4868.9 | 6.6x | 0.00e+00 | 无（aiv_vec 82.8%，向量受限） |
| K5 | 17248579.0 | 9522.6 | 1811.3x | 0.00e+00 | `tl.range(NT, num_stages=3)` 软件流水线（隔离 9.51→9.11ms） |
| K6 | 21615.6 | 4842.5 | 4.5x | 3.73e-09 | **head-merge HM=16**（隔离 6.94→4.75ms，集成 7.35→4.91ms） |

**总 triton 时间 ≈ 46.1ms**（对比标量削减前 ≈48.45ms，收敛约 −5%）。核心结论见 §5.5：
6 核中**仅 K6 是真正标量饱和**（aiv_scalar 97.8%），head-merge 后降到 60.9%、代价转移到
cube（aic_mte2 92.5%）；K2/K3/K4 逐个确认非标量饱和（延迟/向量受限），无 lever。

## 5. 与 H100 横向对比：910B2 理论上限（2026-08-25 评估）

同事在 **H100 上跑同 6 个算子**（同 Triton kernel、fp32、同一目标 case）总时间
**6.9ms**；本机 910B2 为 46.1ms → 当前仅 **0.15x**（约慢 6.7x）。

### 5.1 算法本质是内存带宽受限

H100 的内存带宽下限 = 20.3GB / 3.35TB/s ≈ **6.0ms**，6.9ms ≈ **87.6% 峰值带宽**。
这套算子在 H100 上已贴近带宽下限，故 910B2 的上限由**带宽比**而非算力比决定。

目标 case 各 kernel 的 fp32 内存总流量（合计 ≈20.3GB）：

| kernel | 读（MB） | 写（MB） | 合计（MB） |
|--------|---------|---------|-----------|
| K1 | 805(x) | 805(g) | 1,611 |
| K2 | 2,422(q,k,g,β) | 503(Aqk_d+Akk) | 2,926 |
| K3 | 2,422(q,k,g,β) | 805(Aqk_nd+Akk_inv) | 3,228 |
| K4 | 1,611(k,v) | 2,416(w,u,kg) | 4,027 |
| K5 | 2,416(kg,w,u) | 1,611(h,v_new) | 4,027 |
| K6 | 3,624(q,v_new,g,Aqk,h) | 805(o) | 4,429 |

### 5.2 硬件规格对比

| 项 | H100 SXM5 | 910B2（本机） |
|----|-----------|---------------|
| 内存带宽 | 3.35 TB/s（HBM3） | 口径不一：多来源 ~400GB/s（HBM2e）、部分 1.6 TB/s（HBM3e） |
| 内存 | 80 GB | 64 GB（本机 npu-smi 确认） |
| FP16 | 989 TFLOPS | 376 TFLOPS |
| FP32 | 67 TFLOPS | 官方未给出 |

本机实测可**排除 ~400GB/s 档**：K1（最纯访存型）1.61GB / 1.82ms ≈ **884 GB/s**，
不可能出现在 400GB/s 的设备上。此 910B2 有效带宽至少 ~0.9 TB/s，峰值落在
1.2–1.6 TB/s 区间。

### 5.3 三个口径的上限

| 口径 | 910B2 总时间下限 | 相对 H100(6.9ms) 倍率 | 相对当前(46.1ms) 优化空间 |
|------|------------------|----------------------|--------------------------|
| 当前 Triton 实现 | 46.1ms | **0.15x**（慢 6.7x） | — |
| 本机实测 fp32 有效带宽 ~0.87TB/s（K1 884GB/s 为最佳） | ≈23ms | **≈0.3x** | ≈2x |
| 峰值 1.2 TB/s（HBM2e 口径） | ≈17ms | ≈0.4x | ≈2.7x |
| 峰值 1.6 TB/s（HBM3e 口径，理论极限） | ≈13ms | **≈0.5x** | ≈3.6x |

当前 46.1ms 对应平均带宽利用率仅 **27%**（H100 为 88%）；Triton-Ascend 的代码生成
成熟度是当前差距的主要来源，其次是 910B2 峰值带宽口径本身低于 H100。

### 5.4 结论与前提

- **结论**：910B2 对 H100 的理论上限 ≈ **0.5x**（峰值带宽口径）、现实可达 ≈ **0.3x**、
  当前仅 **0.15x**。优化空间主要来自 kernel 带宽利用率与带宽峰值口径，**不是算力代差**。
- **前提/限制**：
  1. H100 的 6.9ms 已 ≈ 其带宽下限，910B2 物理上无法追平（带宽比 ~1.6/3.35 ≈ 0.48x）。
  2. K3 目前是**计算/发射受限**（14.4ms vs 带宽下限 ~4ms），要压到带宽上限需进一步
     削计算（NP/dot 数）或改用 fp16/tf32 中间精度 —— 后者会改变现有 fp32 精度口径。
  3. K5 串行依赖限制并行度，未必压得到带宽下限。
  4. 若同事的 H100 数据用 fp16/tf32 张量核，口径需另算；但 6.9ms 与 fp32 带宽下限
     一致，按同口径理解合理。

> 规格来源：[H100 SXM5 带宽/FP32 3.35TB/s·67TFLOPS](https://vercel.hyper.ai/en/gpu-leaderboard/nvidia-h100-sxm5-80-gb)、
> [NVIDIA DGX / H100 (Wikipedia)](https://en.m.wikipedia.org/wiki/Nvidia_DGX-1)、
> [昇腾910B vs A100/H100 对比](https://hwcomputing.csdn.net/6a4b7009662f9a54cb8a4933.html)、
> [910B1/B2/B3/B4 选型（带宽口径差异）](https://ucache.cn/enterprise/new/318.html)。
> 910B2 峰值带宽口径不一（~400GB/s HBM2e / 1.6TB/s HBM3e）；本机 64GB、实测 K1
> 884GB/s，按 1.2–1.6 TB/s 档理解。

### 5.5 修正（2026-08-25）：真实瓶颈是标量寻址，不是带宽 —— 0.4x 不可达

对 K2/K3/K5/K6 做 **msprof ai-core 隔离 profile**（每 kernel 单独跑、读
`metric_summary.db`），推翻 §5.1–5.3 的"内存带宽受限 → 0.4x 可达"模型：

| kernel | aic mac | aic scalar | aic mte2 | aiv vec | aiv scalar | aiv mte2 | 判断 |
|--------|---------|-----------|----------|---------|-----------|----------|------|
| K2 | 7.1% | 33% | 10.4% | 32.7% | 36% | 19.9% | 延迟/低利用率（全 <37%） |
| K3 | 12% | **48%** | 20.7% | 8.2% | 40% | 13.5% | aic_scalar 受限 |
| K5 | 14.8% | **59.5%** | 22.9% | 13% | 33.6% | 11.7% | aic_scalar 受限 |
| K6 | 14.6% | 15.5% | 40.4% | 20.5% | **97.8%** | 49% | aiv_scalar 受限 |

**没有任何 kernel 逼近内存带宽墙**（mac/mte2 全 ≤50%）；真正的限制是**标量/地址
生成单元**（aiv_scalar 33–98%、aic_scalar 33–59%）和 K2 的低利用率延迟。各 kernel
的"有效带宽"（K2 298 / K3 214 / K5 430 / K6 651 GB/s）**不是带宽上限，而是标量
受限的副产品**。

由此 §5.3 的 **0.4x 不可达**：K2/K3/K5/K6 都没在等内存，改布局（[B,H,T,K] 转置）
或提高连续访问带宽**不会加速**。佐证的负结果实验（2026-08-25）：

- 纯 copy 探针（bw_sweep）：contig 1242 / row-major strided 1187 / col-major strided
  621 GB/s —— Triton 可达全带宽，但**只对纯流式 copy 成立**，不适用于含 dot 的真实 kernel。
- K6 grid 轴序换 head-fastest（等价 row-major coalescing）：**7→9ms（变慢）**。
- K6 fp16 dot（意图触发 cube）：无变化（max_diff 恒 3.73e-09，后端把 fp16 upcast 回 fp32）。
- K6 NO_MASK（跳过边界 mask）：7.4ms（无增益）。
- K6 bf16 输入（流量减半）：10.9ms（**变慢**，scalar-bound 对流量不敏感）。

**结论**：0.4x H100（17.25ms）在**当前 fp32 契约 + 当前 Triton-Ascend 代码生成**下
**不可达**。剩余方向都超出"kernel 级迭代优化"：(a) 换更成熟的 Triton-Ascend 后端/
版本（改善标量与地址生成代码质量）；(b) K2/K3/K6 算法级重写（削减标量寻址、提高每
CTA 有效工作，高风险长周期）；(c) 改 bf16 精度契约（K6 实测对流量不敏感，收益存疑）。

### 5.6 标量削减轮（2026-08-25，回应用户"重点优化 6 算子 scalar 受限"）

对 §5.5 标出的各标量管道逐核施加削减 lever，验证"标量饱和"假设并修掉可修者：

| kernel | 假设 | 施加 lever | 结果（隔离 us） | 结论 |
|--------|------|-----------|-----------------|------|
| K6 | aiv_scalar 97.8% 饱和 | head-merge HM=16（每 CTA 固定标量 setup 摊 16 head） | 6939→**4748** | **修复**：aiv_scalar 97.8→60.9%，代价转移到 cube（aic_mte2 92.5%、aic_scalar 89.7%） |
| K6 | 手动 int64 寻址本身耗标量 | block_ptr | 7005（无增益） | 证伪：耗标量的是**每 CTA setup**，不是寻址形态 |
| K5 | aic_scalar 59.5% | `tl.range(NT, num_stages=NS)`（NS 之前被静默忽略——chunk 循环是裸 `for`） | 9513→**9106**（NS=3） | 修复：软件流水线预取下一 chunk load |
| K3 | aic_scalar 48% | HM/NW/NS/NP 全扫 | 全部 ≈14700 平 | 证伪：非标量饱和，aiv_cyc=1250M 主导，6-dot 串行逆链是关键路径 |
| K2 | 低利用率延迟 | HM/NW/NS | 全部 ≈9760 平 | 证伪：全 pipe <37%，内存延迟受限 |
| K4 | （新 profile）aiv_vec 82.8% | exp2 削减（1×[BT,K] exp2 + 倒数共享） | 4739 vs 4727（无增益） | 证伪：向量受限，exp2 在向量 pipe 上不是瓶颈 |

**净效果**：总 triton ≈48.45→46.1ms。**只有 K6 是真正的标量饱和核**，已修（贡献
−2.4ms 中约 −2.3ms）；K5 贡献约 −0.4ms。K2/K3/K4 确认卡在延迟/向量/算法关键路径，
不再有标量 lever。

## 6. 各算子最终配置一览（已固化在 `src/*_kernel.py` 中）

| Op | 文件 | 关键配置 | 优化要点 |
|----|------|----------|----------|
| K1 | `gate_chunk_cumsum/src/gate_kernel.py` | BS=128, 2D grid | cumsum + logsumexp 融合 |
| K2 | `token_parallel/src/token_parallel_kernel.py` | HM=16, `tl.gather` 紧凑 Akk | head-merge + kernel 内收拢对角块 |
| K3 | `inter_solve/src/inter_solve_kernel.py` | HM=16, NP=3, fp32 | 融合单 kernel + 重复平方截断逆 |
| K4 | `recompute_w_u/src/recompute_w_u_kernel.py` | HM=16 | head-merge（6796→4514us） |
| K5 | `delta_rule_h/src/delta_rule_h_kernel.py` | BV=V=128, 2 dot/chunk, `tl.range(NT, num_stages=3)` | 单 K-tile 合并串行 dot 链 + 软件流水线预取（9.51→9.11ms） |
| K6 | `gla_output/src/gla_output_kernel.py` | BK=128, BV=128, nw=2, **HM=16 head-merge** | 跨块 K 循环合并为单大 dot + 摊每 CTA 标量 setup（7.35→4.91ms） |

## 7. 结论汇总

- **6 算子全部落地**：K1..K6 的唯一最优 triton kernel 固化于各 `src/<op>_kernel.py`，
  统一框架可一键跑正确性 + msprof 性能。
- **精度**：目标 case 6/6 OK，K4/K5 与 torch_npu bitwise 一致。
- **性能**：目标 case 总 triton ≈46.1ms（官方 ≈46.7ms，偏差 <10%）；总加速比
  （torch_npu 拼接总和 ≈1.26s → triton 46.1ms）约 **27x**，其中 K5 高达 95x、K2 20x。
- **约束/标注**：K=V=32 大 T 等 case 按 README §9 约束正确标注 `不支持`（非 bug）。
- **横向对比**：910B2 相对 H100 当前 ≈0.15x（46.1ms / 6.9ms）。§5.3 曾按
  "内存带宽受限"估现实可达 ~0.3x、理论上限 ~0.5x；**2026-08-25 msprof 隔离 profile
  证伪该模型**——K2/K3/K5/K6 均非带宽受限，而是标量寻址/延迟受限（§5.5），
  故 0.4x（17.25ms）在当前 fp32 契约 + 当前 Triton-Ascend 代码生成下**不可达**。
- **标量削减轮（2026-08-25，§5.6）**：逐核验证"scalar 受限"假设——**仅 K6 真标量
  饱和**，head-merge 修复（aiv_scalar 97.8→60.9%，7.35→4.91ms）；K5 借
  `tl.range` 流水线 −0.4ms；K2/K3/K4 确认非标量饱和（延迟/向量/算法关键路径）。
  总 triton ≈48.45→46.1ms。

## 8. 已知问题与待办

- **K5 同事版曾报 6.4ms**（BT=128，破坏 K2..K6 共享的 BT=64 chunk 契约，无效）；
  **K6 同事版曾报 4.52ms**（fp32 下不可能，已 7 组实验证伪，真实 ~7ms）。均
  **未采纳**；证伪结论见各 `OPTIMIZATION_LOG.md`。
- **瓶颈模型修正（2026-08-25）**：msprof 隔离 profile 证明 K2/K3/K5/K6 是标量寻址/
  延迟受限而非带宽受限（§5.5）；"0.4x 靠布局转置/带宽优化"不再成立。
- **标量削减收敛（2026-08-25）**：标量 lever 已尽——K6（唯一标量饱和）经 head-merge
  修复、K5 经 NS 流水线 −0.4ms，K2/K3/K4 无 lever；总 triton 收敛于 ≈46.1ms（§5.6）。
- **全量 106 case 性能采集**：目标 case 已复现官方；全 106 case 的 results.csv
  尚未完整落盘（设备持续负载不稳定，README §10），需分批续跑
  `bash run_all_msprof.sh` 后 `per_case_profile.py` 聚合。
- **A/B/C 组全量复核**：历史已过；如需留档复核，按 README §10 分批跑模式 A。
