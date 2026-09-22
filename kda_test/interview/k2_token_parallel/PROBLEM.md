# 问题 K2 · token_parallel

> 目录文件：`token_parallel_kernel.py` 里有 torch 参考 `token_parallel_torch` 与待改写/待优化的
> triton kernel `token_parallel_triton`。你只改 kernel；`test.py` 负责比对与计时。

## 0. 算子描述

序列按 64 切 chunk、chunk 内再按 16 切窗口。对每个 16-token 窗口，算窗口内 16 行 × 16 列
的因果打分表，用两种口径写两个输出：**Aqk**（query×key，留对角线）、**Akk**（(k·β)×k，
去对角线）。窗口之间互相独立。

## 1. 输入

| 张量 | shape | 含义 |
|---|---|---|
| `q` | [B, T, H, K] | 每个 token 的 query 向量 |
| `k` | [B, T, H, K] | 每个 token 的 key 向量 |
| `gk` | [B, T, H, K] | 每个 token 逐通道的 gate（log2 空间），控制衰减 |
| `beta` | [B, T, H] | 每个 token 一个标量门 |
| `scale` | float | 常量 `1/sqrt(K)` |

维度：`B`=batch；`T`=token 序号 0..T−1；`H`=head；`K`=特征通道（q/k/gk 最后一维的长度）。

## 2. 输出

| 张量 | shape | 含义 |
|---|---|---|
| `Aqk` | [B, T, H, 64] | token 一行 64 列 = chunk 内 64 个 key 的位置 |
| `Akk` | [B, T, H, 16] | token 一行 16 列 = 窗口内 16 个 key 的紧凑位置 |

## 3. 计算（先划 tile → 单 tile 内全部计算）

> 记号：`BT=64`、`BC=16`；token `t` 的窗口起点 `s=(t//16)*16`、窗口在 chunk 内的列起点
> `w=s%64`。行 token 只对**同一窗口内**"自己及更早"的列 token 打分（因果），故每张表只有
> 对角线上 4 个 16×16 下三角块非零。`⟨·,·⟩`=K 维内积；`⊙`=逐元素乘；`exp2(v)=2**v`。

### 划 tile：每 CTA 一个 (chunk, HM 个头) 的整块

输出 `Aqk[B,T,H,BT]`（行 t 的 64 列 = chunk 内 64 个 key 位置）与 `Akk[B,T,H,BC]`（行 t
的 16 列 = 窗口内 16 个 key 的**紧凑**位置），(chunk, head) 互相独立。收敛 kernel 一个
CTA 处理**整 chunk 的 64 行 × 全部 K 通道**、循环 `HM` 个头（目标 HM=16）：

```
grid = (cdiv(T,BT), B·(H//HM))     # → (NT, B·H//16)；tile = (chunk, HM 个头)
一次载整 chunk q/k/g/β [BT,K]，算两张 [BT,BT]，块掩码后满宽 / 紧凑写回
```

### 单 tile 公式（tile = (b,h) 的一个整 chunk）

```
拆指数 exp2(g[i]-g[j]) = exp2(g[i])·exp2(-g[j]) ⇒ 行/列因子预乘：
qe = q·exp2(g)·scale    ,  ke = k·exp2(-g)          # 各 [BT,K]
Aqk_full = qe @ keᵀ      → 保留 (对角 16 块) ∧ (块内 j≤i)      # 含对角
Akk_full = ((k⊙β)⊙exp2(g)) @ keᵀ  → 保留 (对角 16 块) ∧ (块内 j<i)   # 去对角
```

- Aqk/Akk 差别只在**行向量**（Aqk 用 `q`，Akk 用 `k·β`）与含/去对角线；两路共用 gated-key
  列 `ke`。
- 其余块/元置 0。Aqk 满宽写回 `[B,T,H,64]`；Akk 把每行的对角段收拢成 16 列，紧凑写回
  `[B,T,H,16]`。

### tile 代码（= kernel 主体；HM 头循环，注释即全部逻辑）

```
i_cg, i_hg = tl.program_id(0..1)      # (chunk, head 组)；解码 i_b、chunk 起点
载整 chunk: qc,kc,gc [BT,K]（越界行补 0）、betac [BT]
eg = exp2(gc) ;  eneg = exp2(-gc) ;  ke = kc * eneg       # gated-key 列（共用）
Aqk_full = tl.dot(qc * eg * scale, tl.trans(ke))          # [BT,K]@[K,BT] → [BT,BT]
Akk_full = tl.dot((kc * betac[:,None]) * eg, tl.trans(ke))
keep   = (块号 r//BC == 块号 c//BC) & (行内 r%BC ≥ 列内 c%BC)   # 对角块 ∧ 下三角含对角
strict = (块号 r//BC == 块号 c//BC) & (行内 r%BC > 列内 c%BC)   # 去对角
Aqk_full = tl.where(keep,   Aqk_full, 0.0)
Akk_full = tl.where(strict, Akk_full, 0.0)
tl.store(Aqk + 行 t 写宽 BT, Aqk_full)                    # 满宽 [BT,BT]
tl.store(Akk_scratch + 行 t 写宽 BT, Akk_full)            # Akk 满宽写 scratch
# driver 收拢: Akk = torch.gather(Akk_scratch, 对角段列下标)   # 紧凑 [B,T,H,16]
```

> 收敛 kernel 把同一 chunk 的 4 个 16 窗口拼成一次整 chunk `[64,K]@[K,64]` 大 dot（比逐
> 窗口少启几次小 dot），再整体按"对角 16 块 ∧ 块内因果"置零 —— 与逐窗口数值一致。
> Ascend MTE 列须单调：Akk 的紧凑列映射（对角段在每行内的起始列不同）若窄宽直写会
> 非单调 ⇒ 收敛版（hm2）先满宽写 scratch（按 NT·BT 补齐、store 不加 mask），再在
> driver 里 `torch.gather` 收拢对角段。kernel 内 `tl.gather` 一步收拢（hm3）曾尝试
> （省 driver 侧 ~2ms/调用），但 triton-ascend 3.2.1 对 **tl.dot 输出**做 gather
> 数值错误（实测 Akk max_diff≈0.32）且 ~3× 慢 ⇒ 已回退 hm2 并标注 DEPRECATED
> （详见 §5）。

## 4. 约束与验收

- 评分 case：`B=1, T=16384, H=96, K=128`（T%64==0）；输入输出 fp32。
- 正确性：与 `token_parallel_torch` 逐元素差 < `1e-2`（当前 ~1e-7）。
- **不许改默认 `chunk_size=64 / sub_chunk_size=16`**（破坏上游契约 = 无效解）。
- 尾 chunk 不满 64 时参考会补 0 再裁，真实 token 值不受影响。

```bash
source ../env.sh                        # 容器环境（可 `source ../env.sh 4` 固定到空闲卡）
python3 test.py                         # ① 正确性：PASS + max_diff
msprof --output=./prof_k2 --application="python3 test.py --perf --repeats 7 --warmup 3" \
    && python3 test.py --report ./prof_k2        # ② 基线（msprof Task Duration 每调用均值）
```

官方基线 ≈ **5.55 ms/调用**（±10%，2026-09-07 本机 triton-ascend 3.2.1 msprof 实测口径）；
门槛 `max_diff < 1e-2`。

## 5. 设计创新点与深度解析

> 本节复盘**基线 kernel 的设计决策链**（数字为迭代环境实测，量级参考；随工具链漂移）。
> 本 kernel 是"数学变换 > 编译器技巧"最典型的样本。

- 【数学变换（Route A）】`exp2(g[i]−g[j]) = exp2(g[i])·exp2(−g[j])`：把逐 token 对的
  指数差拆成行/列因子**预乘**，整个 sub-chunk 从"逐元素指数 + 内积循环"变成一次批量
  `tl.dot` —— Python 内层 for j 循环被结构性消除，`aiv_scalar_ratio` 0.38→0.059
  （首次低于 0.10 门限）。代数重组的效果是编译器调度追不上的。
- 【一次 dot 出两张表】Aqk/Akk 只差行向量（q vs k·β）与含/去对角线：共用 gated-key 列
  `ke = k·exp2(−g)`，同一 `[BT,K]@[K,BT]` 大 dot 的产物按两套掩码写两个输出 —— 行
  因子的一次预乘同时服务两路。
- 【CTA 粒度 = 标量放大器】每 CTA 的固定标量 setup（arange 偏移/掩码/指针 base）与
  tile 大小无关：head-merge HM=16 把 24576 个 CTA 并成 1536 个（每 CTA 串行 16 个
  head），10.5→8.66ms；顺带把高 CTA 数触发的 CANN UB 分配 aicore exception 一起规避。
- 【标量裁剪清单】scale 折叠进 pre-dot 的 q；eg/eneg 每元素只算一次；因果掩码提到
  循环外预计算；K 为 2 幂时整条 K 维掩码去掉（constexpr 分派）——每一处都在削
  "每 CTA 固定成本"。
- 【MTE 单调列约束 → 两步写】Akk 对角段在每行的起始列不同，紧凑窄宽写会令 MTE 列
  寻址非单调而越界 ⇒ 收敛版先满宽写 scratch（按 NT·BT 补齐、store 不加 mask），再由
  driver `torch.gather` 收拢。布局妥协换 kernel 内的确定性与免 memset。
- 【工具链实测与回退（重要教训）】kernel 内 `tl.gather` 一步收拢（hm3）曾在旧工具链省
  ~2ms/调用；但 triton-ascend 3.2.1 对 **tl.dot 输出**做 gather 会**静默算错**
  （Akk max_diff≈0.32，非编译失败！）且 ~3× 慢 ⇒ 回退 hm2 满宽写 + driver gather
  （max_diff≈8.9e-8）并保留 DEPRECATED —— "能编译 ≠ 正确"，任何 kernel 级技巧落地
  都必须带精度门槛验证。
- 【收敛论证参考】msprof 各 pipe 利用率全 <37%、cube 仅 ~7%，属内存延迟/低利用率型
  瓶颈；HM/NW/NS 全扫 ≈9.76ms 平台期 —— 声称"参数级杠杆已耗尽"需要这种全扫证据链，
  而不是感觉。
