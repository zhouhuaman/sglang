# 问题 K6 · gla_output

> 目录文件：`gla_output_kernel.py` 里有 torch 参考 `gla_output_torch` 与待改写/待优化的
> triton kernel `gla_output_triton`（当前为收敛版）。你只改 kernel；`test.py` 负责比对
> 与计时。

## 0. 算子描述

KDA 的最后一个 kernel，把两种信息相加得到最终输出：**跨块历史**（当前 token 对"本 chunk
之前所有 chunk 压缩出来的状态"的查询）与**块内精确注意力**（当前 chunk 内、带因果的
token-to-token 注意力）。对每个 (batch, chunk, head)：历史部分用 `q·exp2(g)` 乘状态快照
`h`；块内部分用因果掩码后的 `Aqk` 乘 `v_new`。两条都是矩阵乘，chunk/head 间互相独立。

## 1. 输入

| 张量 | shape | 含义 |
|---|---|---|
| `q` | [B, T, H, K] | 每个 token 的 query 向量 |
| `v_new` | [B, T, H, V] | 修正后的 value（K5 的残差 v_new） |
| `g` | [B, T, H, K] | K1 产出的累积门控 `gk`（log2 空间） |
| `Aqk` | [B, T, H, 64] | chunk 内因果注意力权重（K2/K3 产物）；行 t 的 64 列 = chunk 内 64 个 key 位置 |
| `h` | [B, NT, H, V, K] | K5 输出：每个 chunk 起始的压缩状态快照 `state` |
| `scale` | float | 常量 `1/sqrt(K)` |

维度：`B`=batch；`T`=token；`H`=head；`K`=query 通道；`V`=value 通道（目标 case K=V=128）。
`NT = T/64`；token `t` 属于 chunk `c = t//64`，块内位置 `i = t mod 64`。

## 2. 输出

| 张量 | shape | 含义 |
|---|---|---|
| `o` | [B, T, H, V] | 最终输出，每个 token 一个 V 维向量 |

## 3. 计算（先划 tile → 单 tile 内全部计算）

> 记号：`BT=64`；token `t` 属 chunk `c=t//64`、块内行 `i=t%64`、`tc=64·c`。
> `h_c = h[b,c,h]`（[V,K]）是 chunk c **起始**的压缩状态快照（K5 产物，只含本 chunk
> 之前的信息 → 天然 causal）；`A_c = Aqk[b,tc:tc+64,h]`（[BT,BT]，行 i、列 j = chunk 内
> 位置 j）是块内因果打分（K2/K3 产物，已带 scale）。`tril(A)`=上三角（j>i）清零、含对角。
> `@`=矩阵乘；`⊙`=逐元素乘。两路都是矩阵乘、逐 (chunk, head) 独立。

### 划 tile：每 CTA = 一个 (chunk, V-slab) × HM 个头

输出 `o[B,T,H,V]` 每 (batch, chunk, head) 独立。收敛 kernel 一个 CTA 负责 chunk 的
`BT` 行 × V 的一个 slab `BV`，并循环 `HM` 个头（目标 BV=128 ⇒ V-slab 就是整条 V、
HM=16）：

```
grid = (cdiv(V,BV), NT, B·(H//HM))    # → (1, NT, B·H//16)；tile = (chunk, V-slab)
program_id(0)=V-slab、 (1)=chunk、 (2)=联合 head 组；每 tile 一个 [BT,BV] 累加器
```

### 单 tile 公式（tile 内两路相加 = 最终输出）

一个 tile 要算 chunk 内 64 行的 V-slab 输出，两路累进**同一个 fp32 累加器**：

```
o_cross = ( q ⊙ exp2(g) · scale ) @ h_cᵀ     # ① 跨块：query 读压缩历史  [BT,BV]
o_intra = tril(A_c) @ v_new_c                 # ② 块内：chunk 内 ≤i 精确注意力 [BT,BV]
o_c     = o_cross + o_intra                   # ③ 相加 = 最终输出
```

- ① query 先按自己 token 的逐通道 gate 衰减、再乘 scale（≈ 自己"此刻"的强度），去点乘
  state 的第 v 行 —— 压缩历史里"所有更早 chunk"的记忆；h_c 只有 chunk c 之前的信息 ⇒
  因果已内建、无需掩码。
- ② 只加 `j≤i`（本 chunk 内的历史/自己）；`tril(A_c)` 把 A_c 的 j>i 位置清 0（当前 token
  不看同 chunk 里更晚的 token）。行 i 的因果掩码用 `tl.where(r>=c, A, 0.0)`（值置 0 而非
  store-mask；Ascend MTE 列须单调）。
- ③ 两路权重不可省；无中间 `qg`/`o_cross` 张量 —— 全在寄存器累加器里合并，整块一次写回。

### tile 代码（= kernel 主体，注释即全部逻辑）

```
i_v, i_tg, i_hg = tl.program_id(0..2)     # (V-slab, chunk, head 组)；解码 i_b/hg0
r = tl.arange(0, BT); c = tl.arange(0, BT)
for hh in tl.range(HM, num_stages=NS):    # 头合并循环（HM=16）；head 无关量已提出循环外
    b_o = tl.zeros([BT, BV], fp32)        # 两路共用累加器
    b_q = load q  [tc:tc+BT, :]      ; b_q = b_q * scale            # [BT,K]
    b_g = load g  [tc:tc+BT, :]                                      # [BT,K]
    b_qg = b_q * exp2(b_g)                                           # [BT,K]
    b_h  = load h  [c, i_v*BV:(i_v+1)*BV, :]                         # [BV,K]
    b_o += tl.dot(b_qg, tl.trans(b_h))     # ① [BT,K]@[K,BV] 跨块
    b_A = load Aqk [tc:tc+BT, :BT]                                    # [BT,BT]
    b_A = tl.where(r[:,None] >= c[None,:], b_A, 0.0)   # ② 下三角因果（含对角）
    b_v = load v_new [tc:tc+BT, i_v*BV:(i_v+1)*BV]                   # [BT,BV]
    b_o += tl.dot(b_A, b_v)               # ② [BT,BT]@[BT,BV] 块内
    store o [tc:tc+BT, i_v*BV:(i_v+1)*BV] = b_o   # ③ 两路已在累加器相加
```

> 收敛 kernel 把与 head 无关的因果 mask / 边界 mask / 偏移向量提升到头循环外（标量寻址
> 削减，目标 case 6.94→4.75ms）。K 维不切小 dot（driver 用 `BK=min(K,128)`；目标 K=128 →
> 单条 [BT,K]@[K,BV] dot）。尾 chunk 不满 BT / 尾 V-slab 不满 BV：行/列 mask 置 0、因果
> mask 把越界行也清零，不污染累加器（参考同样补 0 再裁）。

## 4. 约束与验收

- 评分 case：`B=1, T=16384, H=96, K=V=128`（NT=256，T%64==0）；输入输出 fp32。
- 正确性：输出 `o` 与 `gla_output_torch` 一致，逐元素差 < `1e-2`（当前 ~1e-7）。
- **不许改默认 `chunk_size=64`**（破坏上游契约 = 无效解）。`Aqk`/`h`/`v_new`/`g` 都是
  给定输入（上游 kernel 产物），不得跳过 ①/② 任何一路或改变两路权重。

```bash
source ../env.sh                        # 容器环境（可 `source ../env.sh 4` 固定到空闲卡）
python3 test.py                         # ① 正确性：PASS + max_diff
msprof --output=./prof_k6 --application="python3 test.py --perf --repeats 7 --warmup 3" \
    && python3 test.py --report ./prof_k6        # ② 基线（msprof Task Duration 每调用均值）
```

官方基线 ≈ **4.53 ms/调用**（±10%）；门槛 `max_diff < 1e-2`。
