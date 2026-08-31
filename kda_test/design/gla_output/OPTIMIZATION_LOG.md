# OPTIMIZATION_LOG — gla_output (Kernel 6 / GLA Output)

## 任务信息

| 字段 | 值 |
|------|-----|
| 算子名称 | gla_output (chunk_gla_fwd_o_gk) |
| 任务模式 | optimize-existing-kernel |
| 原始 kernel | `src/gla_output_kernel.py` |
| 目标硬件 | Ascend 910B2 |
| 环境 | triton-ascend-env-zhm, triton 3.2.1, CANN 9.0.0 |
| 完成时间 | 2026-08-18 |

---

## 计算内容

```
o[t] = o_cross[t] + o_intra[t]

跨块 (cross):
    q_gated[t, k] = q[t, k] * scale * exp2(g[t, k])
    o_cross[t, v] = (q_gated @ h^T)[t, v]

块内 (intra):
    A_masked = where(lower_triangular, Aqk, 0)
    o_intra[t, v] = (A_masked @ v_new)[t, v]
```

Tile 参数: `BT=chunk_size(64)`, `BK=32`, `BV=32`, `num_warps=1`, `num_stages=1`

---

## 优化路线总览

| 路线 | 策略 | 文件 | 10/10 | Speedup 范围 |
|------|------|------|-------|-------------|
| **Route A** | `tl.make_block_ptr` + fp32 causal mask | `src/gla_output_kernel_opt_routeA.py` | PASS | 1.39x - 2.34x |
| **Route B** | 手动指针算术 + fp32 causal mask | `src/gla_output_kernel_opt_routeB.py` | PASS | 1.52x - 2.28x |
| **Route C** | 2D grid (NV×NT, B×H) + fp32 causal mask | `src/gla_output_kernel_opt_routeC.py` | PASS | 1.44x - 2.15x |

---

## 路线详情

### Route A: `tl.make_block_ptr` + fp32 causal mask

**改动**:
1. 因果 mask 从 `i32` 升级为 `fp32`: `tl.arange(0, BT)[:, None].to(tl.float32) >= ...`
2. 其余使用 `tl.make_block_ptr` 做 tensor 访问（与原始 kernel 一致）

**分析**: 因果 mask 从 i32 改 fp32 避免 `tl.where` 时的隐式类型转换，但总体性能提升有限。

### Route B: 手动指针算术 + fp32 causal mask

**改动**:
1. 替换所有 `tl.make_block_ptr` 为手动 base + offset 计算
2. 显式构造 row/col mask 做 boundary check（替代 block_ptr 的 `boundary_check` 参数）
3. K 循环中 q/g/h 的加载全部使用 `tl.arange` 偏移 + `mask` 显式掩码

**核心代码模式**:
```python
# q tile: [BT, BK] — 手动偏移
q_offs = tl.arange(0, BT)[:, None] * s_q_t + tl.arange(0, BK)[None, :]
q_mask = (i_t * BT + tl.arange(0, BT)[:, None] < T) & (i_k * BK + tl.arange(0, BK)[None, :] < K)
b_q = tl.load(q + base + q_offs, mask=q_mask, other=0.0)
```

**分析**: 手动指针算术消除了 `tl.make_block_ptr` 的构造开销，在 Ascend 后端上 `make_block_ptr` 有不可忽略的 IR 膨胀成本。Route B 的 kernel 耗时普遍低于 Route A (~0.286ms vs ~0.320ms)，speedup 提升约 0.05-0.15x。

### Route C: 2D grid 拓扑

**改动**:
1. Grid 从 `(cdiv(V, BV), NT, B * H)` 3D 改为 `(cdiv(V, BV) * NT, B * H)` 2D
2. `program_id(0)` 分解: `i_v = pid(0) % NV`, `i_t = pid(0) // NV`

**分析**: 2D grid 的 CTA 总数与 3D 相同，但合并维度可能导致 NPU 调度器分配不均。tiny_T100 上 speedup 从 Route B 的 2.18x 降至 1.76x，说明 2D 拓扑对 Ascend 调度器不友好。

---

## 优化前后对比 (目标 CASE: tiny_default B=1, T=128, H=2, K=64)

| 指标 | torch_npu baseline | Route A | Route B | Route C |
|------|-------------------|---------|---------|---------|
| triton 耗时 | 0.468ms | 0.319ms | 0.291ms | 0.299ms |
| speedup | 1.00x | 1.47x | **1.57x** | 1.55x |
| max_diff | — | 3.04e-05 | 3.04e-05 | 3.04e-05 |
| 精度 | — | PASS | PASS | PASS |

---

## 核心发现

1. **`tl.make_block_ptr` 有可测量的开销**: Route B (手动指针) 比 Route A (block_ptr) 快约 9-10%，说明 Ascend 后端对 block_ptr 的 IR 降级存在优化空间。
2. **2D grid 无收益**: Route C 合并 V-tile 和 chunk 维度后性能反降，尤其在 T=100 时 speedup 从 2.18x 降至 1.76x。3D grid 让 NPU 调度器有更好的并行度感知。
3. **fp32 mask 是正确选择**: i32 因果 mask 在 `tl.where` 中触发隐式类型转换，fp32 mask 消除此开销。
4. **tiny case 下 torch_npu 启动开销占主导**: 所有 case 的 torch_npu 耗时 ~0.44-0.75ms，而 triton kernel 耗时 ~0.28-0.37ms。torch_npu 元算子链路 (exp2 + matmul + tril) 的 kernel launch 开销在微小 tensor 上不可忽略。
5. **精度一致**: 三个路线的 max_diff 完全相同 (3.04e-05)，因为计算逻辑未变，仅改变了数据加载方式。

---

## 全量回归验证

10/10 case 全部 PASS，所有 case 精度 max_diff < 1.53e-04。

### 逐 case Route B 结果 (最优路线)

| case | shape | torch_npu | triton | speedup | max_diff |
|------|-------|-----------|--------|---------|----------|
| tiny_default | [1, 128, 2, 64] | 0.455ms | 0.291ms | 1.57x | 3.04e-05 |
| tiny_partial_chunk | [1, 63, 2, 64] | 0.677ms | 0.322ms | 2.10x | 3.91e-05 |
| tiny_single_head | [1, 128, 1, 64] | 0.447ms | 0.294ms | 1.52x | 2.57e-05 |
| tiny_H3 | [1, 128, 3, 64] | 0.454ms | 0.288ms | 1.58x | 2.95e-05 |
| tiny_T65 | [1, 65, 2, 64] | 0.657ms | 0.287ms | 2.28x | 2.92e-05 |
| tiny_T96 | [1, 96, 2, 64] | 0.653ms | 0.288ms | 2.26x | 2.58e-05 |
| tiny_T1 | [1, 1, 2, 64] | 0.638ms | 0.286ms | 2.23x | 2.93e-05 |
| tiny_T2 | [1, 2, 2, 64] | 0.628ms | 0.284ms | 2.21x | 2.20e-05 |
| tiny_T100 | [1, 100, 2, 64] | 0.625ms | 0.286ms | 2.18x | 1.53e-04 |
| tiny_T127 | [1, 127, 2, 64] | 0.628ms | 0.286ms | 2.20x | 2.57e-05 |

---

## 最终产出

| 文件 | 路径 |
|------|------|
| Route A kernel | `src/gla_output_kernel_opt_routeA.py` |
| Route B kernel | `src/gla_output_kernel_opt_routeB.py` |
| Route C kernel | `src/gla_output_kernel_opt_routeC.py` |
| Route A test | `test_routeA.py` |
| Route B test | `test_routeB.py` |
| Route C test | `test_routeC.py` |
| 优化日志 | `OPTIMIZATION_LOG.md` |

---

## 结论

**推荐 Route B** (手动指针算术 + fp32 mask)。相比 Route A 的 `tl.make_block_ptr`，手动指针算术在 Ascend 后端上减少了 IR 构造开销，kernel 耗时降低约 9%。Route C 的 2D grid 合并无收益且可能降低调度效率。

所有路线的 speedup 在 1.4x-2.3x 范围内，受限于 tiny case 下 torch_npu 的 kernel launch 开销主导。在更大 batch/token 规模下，triton kernel 的单 kernel 融合优势会更明显。

### 是否收敛

**是** — 已满足收敛条件:
- 所有路线 10/10 PASS
- Route B 在所有 case 上 speedup > 1.5x
- 进一步的 grid 拓扑变化 (Route C) 已验证无收益
- 手动指针算术 (Route B) 已验证优于 block_ptr (Route A)

---

## 第二轮优化: 目标 case 跨块 K 循环合并（BK 32→128，8.2ms → 6.8ms）

### 背景
早期优化针对 tiny case (T≤128, H≤3)。目标 case (B=1, T=16384, H=96, K=V=128) 下
grid = (1, 256, 96) = 24576 CTAs，每 CTA 的跨块路径用 BK=32 的 K 循环做 4 个
串行小 dot（`[64,32]@[32,128]`），加上块内 1 个 `[64,64]@[64,128]` dot。

### 瓶颈诊断（exp_opt_bk.py 隔离扫描）
| 配置 | 时间 |
|------|------|
| BK=32, BV=128, nw=1 (原始) | 8.22ms |
| **BK=64, BV=128, nw=1** | 6.99ms |
| **BK=128, BV=128, nw=2** | **6.84ms** |
| BK=128, BV=128, nw=4 | 6.85ms |
| BV=64 (V 切分) | 12.8ms (差，勿用) |

跨块 4 个串行小 dot 是瓶颈；合并为 1 个 `[64,128]@[128,128]` 大 dot 后
串行链缩短，17% 提速。BV 必须保持 =V（V 切分使 CTA 数翻倍且 per-CTA 开销主导）。

### 改动（`src/gla_output_kernel.py`）
1. 去掉 `@triton.autotune`（原配置锁死 BK=32/BV=128/nw=1），改 driver 显式控制。
2. `BK = 128 if K >= 128 else K`；`num_warps = 2 if BK >= 128 else 1`。
   - 目标 K=128 → BK=128, nw=2；
   - K=64 → BK=64（整 K，避免 BK=128 的零填充浪费）。
3. 正确性：目标 case max_diff=8.94e-08，K=64 小 case 5.96e-08，全部 PASS。

### 结果
- 目标 case：8.20ms（msprof）→ 6.84ms（隔离实验）/ 7.42ms（driver 验证）。
- 官方 msprof 统一 bench 待更新（results.csv）。

---

## 第三轮: 同事 4.52ms 证伪（2026-08-21 ~ 2026-08-24）

同事在 run_custom.py 框架下报 K6=4.52ms。经 **7 组实验**核验（exp_nw 配置扫描、
exp_clock warm 态、exp_msprof 双 kernel msprof、exp_chain chained-vs-random、
exp_dtype 尝试、exp_col 逐 col 隔离），**双方 kernel 结构等价（都是
[64,128]@[128,128] 跨块 + [64,64]@[64,128] 块内 2-dot），耗时全部 ~7ms**。

| 实验 | 同事 | 我们 |
|------|------|------|
| chained 输入 | 7.22ms | 7.04ms |
| random 输入 | 7.11ms | 7.14ms |
| msprof device 时间 | 7.16ms (wall) | 7.19ms (wall), 6.6-6.8ms device |

**结论**: 同事 4.52ms 是 harness/环境产物，非真实加速，**不采纳**。

### 目标 case 内存下界分析（为什么 ~7ms 已到地板）
每 CTA 读 h[128,128]=64KB + q/g/v/A/o ≈ 208KB；24576 CTA 总流量 ~5.1GB。
7.39ms ≈ 700GB/s，贴近 HBM 有效带宽。**fp32 输入下无下探空间**；
只有把输入改 bf16（流量减半）才可能到 ~3.6ms，但那改变精度契约。
官方 msprof 集成 K6 = **7.39ms**（max_diff 3.7e-9）。
---

## 第四轮: 标量寻址瓶颈证伪"内存下界"（2026-08-25）

msprof ai-core 隔离 profile（只跑 K6 kernel，读 metric_summary.db）推翻上一轮的
"~7ms ≈ 700GB/s 贴近 HBM 有效带宽"：

| 指标 | 值 | 含义 |
|------|-----|------|
| aic mac | 14.6% | cube 乘加 15% 忙 → 非算力瓶颈 |
| aic mte2 | 40.4% | cube 侧数据搬运 40% |
| aiv vec | 20.5% | vector 计算 21% 忙 → 非算力瓶颈 |
| aiv scalar | **97.8%** | **vector 标量/地址生成单元饱和 = 关键路径** |
| aiv mte2 | 49% | vector 侧数据搬运 49% |

结论：K6 是 **aiv_scalar（地址生成）受限**，不是内存受限。mte2 仅 ~49%，流量
减半不会加速。

### 负结果实验（均不采纳，恢复基线）
1. **grid 轴序换 head-fastest**（i_bh 最快 → q/g/v/o/A 跨 CTA 连续）：7→9ms（变慢）。
   纯 copy 探针里 row-major=1187 vs col-major=621 GB/s 的差异**不适用于含 dot 的真实 kernel**。
2. **fp16 dot**（意图触发 cube）：max_diff 恒 3.73e-09、耗时不变 → 后端把 fp16
   upcast 回 fp32，cube 未启用。
3. **NO_MASK 特化**（T/K/V 全整除时跳过边界 mask）：7.4ms，无增益。
4. **bf16 输入**（流量减半）：10.9ms（**变慢**），scalar-bound 对流量不敏感。

**K6 收敛值维持 ~7ms（BK=128/BV=128/nw=4）。** 0.4x H100 不可达的根因是标量寻址，
见 `unified/ANALYSIS.md §5.5`。

---

## 第五轮: head-merge 削减 aiv_scalar（2026-08-25，6.94→4.75ms）

第四轮确认 aiv_scalar 97.8% 饱和是 K6 关键路径。诊断：每 CTA（grid = (1, NT, B×H)，
24576 CTA）做一次完整标量 setup（`tl.arange` 偏移、r/c/m_s 因果 mask、q/g/h/v/A 的
pointer base 与边界 mask）。这些 setup 与 head 无关却每 head 重算一遍 → 标量 pipe 饱和。

### 假设验证（`unified/k6_scalar_ab.py` 同进程 A/B）
| 变体 | 做法 | 时间 | 结论 |
|------|------|------|------|
| base（原） | 手动指针，每 CTA 1 head | 6939us | 基线 |
| **hm16** | head-merge：`for hh in tl.range(HM)`，每 CTA 16 head，所有偏移/mask/pointer 向量**提循环外** | **4748us** | **−31.6%** |
| hm32 | 同上，HM=32 | 4743us | 无进一步收益 |
| bp | `tl.make_block_ptr` 重写 | 7005us | **证伪**"手动 int64 寻址耗标量"：耗标量的是每 CTA setup，不是寻址形态 |
| hm16_ns2/ns3 | hm16 + `num_stages` | ~4755us | NS 无增益 |

**关键实现**：`b_o = tl.zeros([BT,BV], float32)` 必须在 `for hh` 循环**体内开头**重声明
（每 head 独立累加器）；r/c/m_s/r_mask/k_mask/v_mask/q_offs/h_offs/v_offs/o_offs/
A_offs/q_mask/h_mask/v_mask2/o_mask 全部 loop-invariant 提循环外。

### 集成（`src/gla_output_kernel.py`）
driver 按 `HM = 16 if (H % 16 == 0 and BV <= V) else 1` 选择；HM>1 走
`chunk_gla_fwd_kernel_o_hm`（grid=(cdiv(V,BV), NT, B*H//HM)，`NS=_K6_NS`,
`num_warps=_K6_NW`），否则回退原 kernel。K6 集成 **7.35→4.91ms**，max_diff 3.73e-09。

### 集成后 msprof 复核
| 指标 | 前 | 后 | 含义 |
|------|-----|-----|------|
| aiv scalar | 97.8% | **60.9%** | 标量 pipe 不再饱和 |
| aic scalar | — | **89.7%** | 新瓶颈：cube 标量/配置 |
| aic mte2 | 40.4% | **92.5%** | cube 侧数据搬运饱和 |

**代价转移到 cube**（aic_mte2 92.5%、aic_scalar 89.7%）：标量问题已解，但 cube 成为
新限制，HM 之上无更多标量 lever。K6 收敛值更新为 **4.91ms 集成 / 4.75ms 隔离**。
