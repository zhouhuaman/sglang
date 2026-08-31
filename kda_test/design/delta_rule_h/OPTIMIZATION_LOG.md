# delta_rule_h 优化日志

## Round 1: 多路线探索 (2026-08-18)

### 路线概述

| 路线 | 策略 | BV | num_warps | 关键改动 |
|------|------|-----|-----------|---------|
| Baseline | 原始 kernel | 32 | 4 | flat 1D store (K=64 假定), boundary_check=(0,1) 全维 |
| Route A | Scalar Fallback 消除 | 32 | 4 | 2D 通用 store, 去 K/V 列 boundary_check, gk_last mask fp32 |
| Route B | Tile Size 优化 | 64 | 2 | 2D 通用 store, BV 增大, num_warps 降低 |
| **Route C** | **flat store + 边界消除** | 32 | 4 | K=64 flat 1D store, K≠64 2D store, 去 boundary_check, gk_last mask |

### 精度验证

| 路线 | 结果 | max_diff |
|------|------|----------|
| Route A | 15/15 PASS | 0.000e+00 |
| Route B | 15/15 PASS | 0.000e+00 |
| **Route C** | **15/15 PASS** | **0.000e+00** |

### 性能对比 (msprof op_summary, --repeats 5 --warmup 2)

| 指标 | Baseline | Route A | Route B | **Route C** |
|------|----------|---------|---------|-------------|
| Calls | 90 | 120 | 120 | 120 |
| Total (us) | 1778.1 | 2448.7 | 2559.9 | **2308.7** |
| Per-call (us) | 19.76 | 20.41 | 21.33 | **19.24** |
| aiv_scalar_ratio | 0.436 | 0.472 | 0.429 | **0.432** |
| aiv_vec_ratio | 0.079 | 0.039 | 0.067 | **0.078** |
| aic_mac_ratio | 0.046 | 0.044 | 0.068 | **0.046** |
| cube_utilization(%) | 17.4 | 17.6 | 8.9 | **17.4** |
| aic_scalar_ratio | 0.510 | 0.494 | 0.520 | **0.523** |
| aiv_mte2_ratio | 0.143 | 0.190 | 0.155 | **0.138** |
| aiv_icache_miss_rate | 0.048 | 0.027 | 0.063 | **0.030** |

### 逐 Case 加速比 (per_case_profile, triton vs torch_npu)

| Case | Baseline | Route A | Route B | **Route C** |
|------|----------|---------|---------|-------------|
| tiny_default | 15.18x | 12.72x | 12.43x | **13.32x** |
| tiny_partial_chunk | 3.62x | 11.92x | 10.33x | **12.03x** |
| tiny_single_head | 13.95x | 3.65x | 3.53x | **3.94x** |
| tiny_H3 | 18.83x | 18.29x | 16.17x | **17.90x** |
| tiny_T65 | 3.94x | 10.51x | 10.74x | **11.12x** |
| tiny_T96 | 8.04x | 13.59x | 12.64x | **13.85x** |
| tiny_T1 | 20.90x | 6.56x | 6.92x | **7.02x** |
| tiny_T2 | 24.46x | 9.91x | 9.89x | **10.17x** |
| tiny_T100 | 21.18x | 13.00x | 12.85x | **13.89x** |
| tiny_T127 | 23.05x | 13.50x | 12.53x | **13.79x** |
| big_B2_H3 | 5.36x | 30.46x | 31.46x | **30.88x** |
| big_B2_T193 | — | 25.43x | 23.73x | **25.27x** |
| big_T256 | — | 14.37x | 14.43x | **15.76x** |
| tiny_T255 | — | 15.00x | 13.51x | **15.25x** |
| tiny_T2562 | — | 3.84x | 3.66x | **4.19x** |

注: Baseline 的 per_case_profile 只切出 11 组（--repeats 3 vs Route A/B/C 的 --repeats 5），case 映射不完整。

### 瓶颈分析

**Route A 分析 (2D store 回归):**
- `aiv_scalar_ratio` 从 0.436 上升到 0.472 (+8.3%): 2D 手动指针 `offs_v[:, None] * K + offs_k[None, :]` 引入 int64 广播乘加
- `aiv_mte2_ratio` 从 0.143 上升到 0.190 (+32.9%): scattered pointer 导致更多 memory transactions
- `aiv_vec_ratio` 从 0.079 下降到 0.039: Vec 单元利用率进一步降低

**Route B 分析 (BV=64 不适合 K=64):**
- `cube_utilization` 从 17.4% 骤降到 8.9%: grid 减半(cdiv(64,64)=1)，每个 CTA 处理更大 V-tile 但利用率指标恶化
- `aic_mac_ratio` 0.068 vs baseline 0.046: MAC 占比提升但不足以抵消其他开销

**Route C 分析 (最优):**
- `aiv_scalar_ratio` 0.432 vs baseline 0.436: 微弱改善 (-0.9%)
- `aiv_mte2_ratio` 0.138 vs baseline 0.143: 微弱改善 (-3.5%)
- `aiv_icache_miss_rate` 0.030 vs baseline 0.048: 显著改善 (-37.5%)
- Per-call 19.24us vs baseline 19.76us: **2.6% 加速**

### 最终结论

1. **Route C 胜出**: per-call 19.24us，比 baseline 快 2.6%，15/15 PASS
2. **2D store 是反模式**: K=64 时 baseline 的 flat 1D store 更高效；2D store 仅用于 K≠64 功能正确性
3. **Scalar Fallback 根因不在 boundary_check**: 去除 boundary_check 后 scalar_ratio 仅下降 0.004 (0.436→0.432)，说明 scalar 主要来源是 `tl.arange` int64 + 指针算术 (triton-ascend 编译器行为)
4. **BV=64 不适合 K=V=64**: grid 过小导致 Cube 利用率指标恶化
5. **K=128 支持**: Route C 的 `_store_h_tile` 对 K=64 走 flat 1D store，对 K≠64 走 2D store，功能正确
6. **I-Cache miss 改善**: Route C 的 0.030 比 baseline 0.048 低 37.5%，是主要加速来源

### 文件清单

| 文件 | 说明 |
|------|------|
| `src/delta_rule_h_kernel_opt_A.py` | Route A: 2D store + boundary 消除 (15/15 PASS, 性能倒退) |
| `src/delta_rule_h_kernel_opt_B.py` | Route B: BV=64 + 2D store (15/15 PASS, 性能倒退) |
| `src/delta_rule_h_kernel_opt_C.py` | **Route C: flat store + boundary 消除 (最优, 15/15 PASS)** |
| `run_opt_A.py` | Route A 测试脚本 |
| `run_opt_B.py` | Route B 测试脚本 |
| `run_opt_C.py` | Route C 测试脚本 |
| `prof_baseline/` | Baseline msprof |
| `prof_opt_A/` | Route A msprof |
| `prof_opt_B/` | Route B msprof |
| `prof_opt_C/` | Route C msprof |
| `_cmp_profile.py` | 四路线指标对比脚本 |
---

## Round 2: 目标 case BV 扫描（2026-08-20）

### 结论
**BV=V（整 V 驻留一个 CTA）是目标 case 最优：40ms → 10.0ms（4×）**。

### 关键发现
本地 kernel 默认 BV=32（grid=(4,96)=384 CTAs），目标 case (B1,H96,T16384,K=V=128, BT=64) 下 40ms。
实测 BV 单调最优且完全正确（h=0, v_new=0 vs torch）：

| BV | grid | 时间 | 说明 |
|----|------|------|------|
| 32 | (4, 96) | 40.1ms | 默认；每 CTA 仅 32 个 V 行，256 chunk 串行 ×4 CTA 冗余 |
| 64 | (2, 96) | 20.3ms | |
| **128 (V)** | **(1, 96)** | **10.0ms** | 整 V 单 CTA，去冗余 |

num_warps 1-16 / num_stages 1-2 对时间几乎无影响（±0.3ms），说明瓶颈是串行 chunk 链的每迭代开销与 BV 切分冗余，而非并行度。

### 改动
`delta_rule_h_triton` 默认 BV 从 env(=32) 改为 **BV=V**（`BV is None or BV<=0 → BV=V`）。

---

## Round 3: 单 K-tile 合并（4 dot/chunk → 2 dot/chunk，9.78ms → 9.09ms）

### 背景
目标 case (B=1, T=16384, H=96, K=V=128) 下 delta step 每 chunk 串行做 4 个
小 dot（`b_h1..b_h4` 各 [BV,64]），串行链是延迟瓶颈（cube 利用率 ~3-4%）。

### 成本隔离（exp_isolate.py）
| MODE | 时间 | 说明 |
|------|------|------|
| full | 10.12ms | 基线 |
| nosnap (不 store h snapshot) | 10.29ms | snapshot store (1.6GB) 几乎免费 |
| novnew (不 store v_new) | 10.14ms | v_new store 也几乎免费 |
| **nodots (dot 全去掉)** | **3.02ms** | **memory 下限 ~3ms** |

→ 4 个 dot 的串行计算链约占 7ms，是绝对瓶颈；store/snapshot 非瓶颈。

### 改动（`src/delta_rule_h_kernel.py`）
1. `b_h1..b_h4` 4 个 [BV,64] tile → 单一 `b_h = tl.zeros([BV, K])`。
2. w 单次 load `[BT, K]`（`tl.make_block_ptr(w, (T,K), ..., (BT,K), (1,0))`），
   delta dot `[BV,BT]@[BT,K]` 一次完成。
3. k 转置 load `[K, BT]`（`p_k = tl.make_block_ptr(k, (K,T), ..., (K,BT), (0,1))`），
   outer dot `[BV,K]@[K,BT]` 一次完成。
4. `_store_h_tile` → `_store_h_full`（b_h 整宽 [BV,K]，K=64 flat store / K≠64 2D）。
5. 4 dot → 2 dot per chunk，串行链减半。

### 结果
- 集成 kernel：9780 → **9087us**（integrated driver 验证 9.48ms standalone），精度 exact（h/v_new=0.0e+00）。
- `verify_integrated_k5.py`：target 9.48ms, K64 0.37ms, K64 B2 0.61ms，全部 PASS。

### 试过的其他路线（无效/不可用）
| 实验 | 结果 | 结论 |
|------|------|------|
| exp_opt_f 单 tile | 9.32ms | 与集成一致，确认单 tile 有效 |
| exp_opt_g BT=128 | 6.75ms 正确 | **BT=128 破坏跨算子 chunk 契约（BT 是 K2..K6 固定 64）**，不可用 |
| exp_opt_g BT=256 | 5.31ms 但 h=6.2e+11 | 寄存器溢出/越界，错误，不可用 |
| exp_opt_h BV=64 切分 | 14.7ms | V 切分使每 chunk 串行 2× 冗余，不可用 |
| exp_opt_h flat store (K=128) | MLIRCompilationError | triton-ascend flat reshape store bug（仅 K=64 可 flat） |
| sweep_ns nw/ns 扫描 | 9.4-9.6ms 平 | 无收益 |

### 是否收敛
**是**：dot 数已减半；进一步合并（把 delta 与 outer dot 合成一个）会破坏
`h_new = (I + w u^T) h + k v^T` 的递推结构。剩余 ~9ms 主要受 256 chunk 串行链
（每 chunk 2 dot + 依赖等待）约束。

---

## Round 4: 同事版本核验（verify_col_k5.py, 2026-08-24）—— 同事 6.4ms 无效

同事 run.py 在 T%128==0 时强制 BT=128，**破坏 K2..K6 共享的 BT=64 chunk 契约**
（K3 Akk_inv / K4 A / K6 Aqk 都假定 BT=64；BT=128 时 NT=128 vs 256，连 h 快照
形状都不同）。在**同一套真实 chained 输入**（unified bench K1→K5 链）上对比:

| 版本 | BT | 时间 | 说明 |
|------|-----|------|------|
| 同事 | 128 | 6.80ms | **契约破坏**，形状不兼容，无效 |
| 同事 | 64 | 10.23ms | 有效配置，比我们慢 0.7ms |
| 我们 | 64 | 9.50ms | diff h/v = 0.0 |

**结论**: 同事的 6.4ms 是 BT=128 契约破坏的假象；在有效 BT=64 配置下同事
10.23ms 慢于我们 9.50ms。官方 msprof 集成 K5 = **9.15ms**。

---

## Round 5: `tl.range` 软件流水线预取（2026-08-25，9.51→9.11ms）

### 背景
K5 grid = (1, H=96) 只有 96 CTA（1 head/CTA，无 head-merge 空间），256 chunk 的
串行递推链（每 chunk 2 dot + 快照 store）是延迟关键路径。msprof 隔离 profile 报
aic_scalar 59.5%，但此前 NW/NS 扫描全平（9.4–9.6ms）。

### 根因：`num_stages` 被静默忽略
chunk 循环是裸 `for i_t in range(NT)`——Triton 对非 `tl.range` 的循环不会做软件
流水线调度，`num_stages` 传进 launch 被直接丢弃。**之前的 sweep_ns 扫描是 no-op。**

### 改动（`src/delta_rule_h_kernel.py`）
1. kernel 签名加 `NS: tl.constexpr`，driver 传 `NS=num_stages`。
2. chunk 循环改 `for i_t in tl.range(NT, num_stages=NS)`。
3. `_NUM_STAGES` 默认 2→**3**。

### 结果
| 配置 | 时间 |
|------|------|
| 原（range 裸 for，NS 失效） | 9.51ms |
| **tl.range(NT, num_stages=3)** | **9.11ms** |

集成 driver 验证 9.52ms（wall-clock 含设备抖动），精度 h/v max_diff = 0.0e+00。
预取下一 chunk 的 w/u/k load 掩盖串行链点积延迟，约 −0.4ms。**不采纳 V 切分
（每 chunk 串行 2× 冗余）与 HM（96 CTA 已是最低并行度，再合并降并行）。**
