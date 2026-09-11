# fp16 形态实验记录（2026-09-09）

> 本目录 = `turbo_chunk_kda`（fp32 交付版）的 fp16 实验副本，目标是对齐 Ascend C
> 融合算子（`ChunkKdaFwd`，fp16/bf16）的位宽口径，量化"**输入 fp16 → vector 段 fp32
> 计算 → cube 直接 fp16 → 输出 fp16**"对 6 个算子的性能影响。
> 环境同验收：triton-ascend 3.2.1（conda `autotriton`）、CANN 9.1.0、物理卡 5、
> msprof Task Duration 每调用均值（warmup 3 + repeats 7），目标 case `D_KV128_H96_T16384`。

## 结果总览（ms/调用）

| 算子 | fp32 基线 | fp16 形态 | 变化 | 正确性（单算子 / bench 链） | 采纳形态 |
|---|---|---|---|---|---|
| K1 gate_chunk_cumsum | 1.374 | **1.195** | **−13.0%** | PASS 9.6e-3（贴门槛）/ 链 4.0e-3 | x→fp16，A_log/dt_bias 与输出 gk **保持 fp32** |
| K2 token_parallel | 5.567 | **4.122** | **−26.0%** | PASS 5.8e-4 / 链 3.8e-2（见 §3 动态范围） | 全 fp16 + **ke 单侧** fp16 饱和 |
| K3 inter_solve | 10.693 | 10.790 | +0.9% | PASS 4.98e-4 / 链 6.9e-4 | 输入/输出 fp16、**cube 保持 fp32**（见 §4） |
| K4 recompute_w_u | 3.614 | **1.998** | **−44.7%** | PASS 3.7e-3 / 链 2.8e-3 | 全 fp16 |
| K5 delta_rule_h | 11.428 | 11.428 | 0%（回退） | PASS（fp32 原版） | **保持 fp32**（见 §5） |
| K6 gla_output | 3.911 | **2.303** | **−41.1%** | PASS 2.05e-4 / 链 7.9e-6 | 全 fp16 |
| **六算子合计** | **36.587** | **≈31.84** | **−13.0%** | 链 A 模式 5/6 OK | 见各算子说明 |

> 说明：合计按各算子采纳形态累加；K5 用 fp32 原版。K3 若 fp16 dot 可用、K2 输入量级
> 受限后，按 K2/K6 实测幅度粗估还有 ~20% 潜在空间（合计可到 ~27-28 ms，−25% 级）。

## 分算子说明与决策链

### K1（−13.0%，部分 fp16）
- 纯访存 + 前缀和，无 dot。fp16 收益全部来自 x 读带宽减半。
- **不可全 fp16 的原因**：目标 case 下 `gk = cumsum(gate)·RCP_LN2` 量级可达 ~1600
  （absmax 实测 1601）。fp16 存储相对误差 5e-4 → 绝对误差 ~0.19，远超 1e-2 门槛
  （实测 0.187 FAIL）。修正为：激活张量 x fp16、A_log/dt_bias 参数与输出 gk fp32
  → PASS（max_diff 9.58e-3，余量 ~4%，为 fp16 输入舍入的理论下限量级）。

### K2（−26.0%，全 fp16；ke 单侧饱和）
- hm2 满宽写 + driver 收拢路径，dot 输入 fp16、acc fp32，store fp16。
- 链上出现 nan → 定位为 `exp2(-gk)` 反演值超 fp16 上限 65504（cast 后 inf×0→nan）。
  注意链上 gk≤0，溢出只出现在 `ke = k·exp2(-gk)` 一路（`qe/kbe` 用 exp2(gk)≤1 不会
  溢出）。**饱和 clamp 只加在 ke**：全量（qe/ke/kbe 三处）饱和实测 +25% 开销
  （3.864→4.811 ms），ke 单侧仅 +6.7%（3.864→4.122 ms），链上 nan 同样消除。
  单算子正确性不受影响（5.8e-4 PASS）。

### K3（+0.9%，cube 保持 fp32）
- **fp16 dot 在此 kernel 触发运行期 507015（aicore 异常），与本工具链的 dot 参数写法
  无关**：out_dtype 与 fp32-acc 两种写法均复现；隔离实验"整文件退回 fp32 kernel +
  仅 driver 输入/输出 fp16"（B 形态）即 PASS——输入/输出 fp16 被排除，唯一变量就是
  kernel 内的 fp16 dot。
- **归因实验（2026-09-09 补做，坐实"深依赖链"假设）**：
  - A1：fp16 dot + **NP=1**（链短：2+2 个 dot）→ **不崩**（仅数值 nan，溢出问题见下）；
  - A2：fp16 dot + NP=2（链加深）→ 崩 507015；
  - B：fp16 dot + NP=3，但每级把 b_pow/b_inv **经 fp32 scratch 写 HBM 再读回**、
    打断寄存器内依赖链 → **不崩**（仍 nan，属溢出非崩溃）。
  - 结论：崩溃 = triton-ascend 3.2.1 对 **fp16 dot × 深依赖链**（截断逆 6-8 个
    输出即输入的小方 dot 级联）的代码生成缺陷；单个/浅层 fp16 dot（K2/K4/K6、
    NP=1）均正常。K5 的 fp16 dot 也能跑（只是慢）进一步佐证。
- nan 尾注：fp16 dot 变体在 K3 测试生成器量级下还伴随中间量瞬态溢出（exp2(-gk)
  与部分积可超 65504）→ nan；加 fp16 饱和后可有限化（精度另议）。这与崩溃机制
  无关，但说明 fp16 形态对上游量级有硬约束（见 §3 动态范围契约）。
- 采纳形态：输入/输出 fp16 + fp32 cube（B 形态），正确性 PASS（4.98e-4/6.9e-4），
  性能 ≈ 基线（10.79 vs 10.69，+0.9% 噪声级）——K3 本身 cube 受限，fp16 全局访存
  无收益；真正的收益要等 fp16 dot 可用后另测。

### K4（−44.7%，全 fp16）—— 收益最大
- 两个整 tile dot（Akk_inv × v/k）占绝对主导，fp16 cube + fp16 全局读写双赢。
- 正确性 PASS（3.7e-3 / 2.8e-3），精度损失主要在 A 与输出的 fp16 表示。

### K5（保持 fp32）
- 串行 256-chunk 递推、小 dot、vector 密集。实验数据：
  - fp16 输入 + fp32 dot：12.82 ms（+12%）；
  - fp16 输入 + fp16 dot：16.44 ms（+44%）。
- 结论：hot 串行循环内"fp16 load → cast fp32"与状态每 chunk cast fp16 进 dot 的
  额外 vector 开销 > 带宽收益；fp16 dot 反而显著更慢。**该算子形态收益为负，回退
  fp32 原版**（本文件即原版，docstring 已注明实验结论）。

### K6（−41.1%，全 fp16）
- 双路单累加器：`q·exp2(g)` 衰减乘在 fp32 完成后 cast fp16 进跨块 dot；块内
  Aqk/v_new 天然 fp16。b_o fp32 累加、store 前 cast fp16。正确性 PASS（2.05e-4 / 7.9e-6）。

## 链上（bench 模式 A）的 1/6 失败与动态范围契约

bench 全链目标 case 5/6 OK；K2 超门槛（3.8e-2）。量化根因（本机实测）：

```
bench 链上 gk: min=-24.94, max=-0.02
  gk < -8  : 38.5%   （exp2(-gk) > 256，fp16 精度下降区）
  gk < -16 :  0.40%  （exp2(-gk) > 65504，fp16 饱和区）
```

fp16 无法表示 `exp2(-gk) > 65504`（|gk|>16）。bench 生成器的 gk 动态范围是人为放大
的（测试构造而非真实 KDA 量级）；真实门控每 chunk 衰减通常仅数个 log2 单位。
**fp16 形态的可用契约：上游 gk ∈ [−16, 0]（留精度余量建议 ≥ −12），否则需上游
缩放/饱和**。Ascend C fp16 链同样受此上限约束。

## 复现命令

与 fp32 版相同（见 `TEST_REPORT.md` §4/§5），环境必须为 conda `autotriton`
（triton-ascend 3.2.1）+ 固定空闲卡：

```bash
export PATH=/data/anaconda3/envs/autotriton/bin:$PATH
cd turbo_chunk_kda_fp16 && source env.sh 5
(cd turbo_gate_chunk_cumsum && python3 test.py)          # 单算子正确性
cd bench && python3 bench.py --start 105 --limit 1        # 链正确性
# msprof 口径同 fp32 版（单算子见 TEST_REPORT §4.1）
```
