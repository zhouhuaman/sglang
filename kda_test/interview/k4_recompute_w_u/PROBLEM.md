# 问题 K4 · recompute_w_u

> 目录文件：`recompute_w_u_kernel.py` 里有 torch 参考 `recompute_w_u_torch` 与待改写/
> 待优化的 triton kernel `recompute_w_u_triton`（当前为收敛版）。你只改 kernel；
> `test.py` 负责比对与计时。

## 0. 算子描述

给定每个 chunk 的 64×64 单位下三角逆 `A`（= Akk_inv，K3 产物，行主序按 token 存放），
把 Key/Value 用 `A` 乘一遍，"解耦"掉 chunk 内的因果依赖，得到可跨 chunk 递推的
`w`/`u`；另把每个 token 的 key 对齐到 chunk 末尾时间戳得到 `kg`。三者都是**逐 (batch,
head) 独立、逐 chunk 独立**的批量矩阵乘 + 逐元素缩放。

## 1. 输入

| 张量 | shape | 含义 |
|---|---|---|
| `k` | [B, T, H, K] | 每个 token 的 key 向量 |
| `v` | [B, T, H, V] | 每个 token 的 value 向量 |
| `beta` | [B, T, H] | 每个 token 一个标量门（gated delta rule 系数） |
| `A` | [B, T, H, 64] | Akk_inv：每 chunk 一个 64×64 单位下三角逆；行 t 存该 chunk 矩阵第 (t mod 64) 行 |
| `gk` | [B, T, H, K] | K1 产出的逐通道累积门控（log2 空间，**已 cumsum**） |

维度：`B`=batch；`T`=token；`H`=head；`K`=key 通道；`V`=value 通道（目标 case K=V=128）。
`NT = T/64`。一个 chunk = `tc..tc+63`，chunk 内行位置 `i = t - tc`。

## 2. 输出

| 张量 | shape | 含义 |
|---|---|---|
| `w` | [B, T, H, K] | 解耦后的 key：`A @ (k·β·exp2(gk))` |
| `u` | [B, T, H, V] | 解耦后的 value：`A @ (v·β)` |
| `kg` | [B, T, H, K] | 时间对齐的 key：`k·exp2(gk_last − gk)`（gk_last = chunk 末 token 的 gk） |

## 3. 计算（先划 tile → 单 tile 内全部计算）

> 记号：`BT=64`；`tc=64·c` 是 chunk 首 token；取该 chunk 一块
> `A_c = A[b, tc:tc+64, h]`（[64,64] 单位下三角逆，K3 产物，行 i = token `tc+i`、列 j =
> chunk 内更早 token）。`@`=矩阵乘；`·`=逐元素乘、按行广播；`exp2(v)=2**v`。视角：
> A_c 的行 i 编码"第 i 个 token 应如何把更早 token 的信息线性组合进来"，乘完把 chunk 内
> 因果耦合摊平，得到可跨 chunk 递推的 w/u；kg 则是把 key 对齐到 chunk 末时间戳。

### 划 tile：每 CTA = 一个 (chunk, HM 个头)，K/V 单 tile

三个输出 `w[B,T,H,K]`、`u[B,T,H,V]`、`kg[B,T,H,K]` 都逐 (batch, chunk, head) 独立。收敛
kernel 一个 CTA 处理**一个 chunk × 整条 K 与整条 V**（`BK=K`、`BV=V` 单 tile 直通，目标
K=V=128 ⇒ 无内维循环），循环 `HM` 个头（目标 HM=16）：

```
grid = (NT, B·(H//HM))     # → (NT, B·H//16)；tile = (chunk c, HM 个头)
一个 CTA 载: A_c[BT,BT]、β_c[BT]、k_c[BT,K]、gk_c[BT,K]、v_c[BT,V]，一次出三输出
```

### 单 tile 公式（tile = (b,h) 的一个整 chunk）

行输入先做逐元素缩放（gate 先把每个 key 衰减到"它实际出现时刻的强度"，A_c 再把它摊进
chunk 内所有更晚 token 的 w 行）：

```
w  = A_c @ ( k_c ⊙ β_c ⊙ exp2(gk_c) )        # [BT,BT]@[BT,K] → [BT,K]  解耦 key
u  = A_c @ ( v_c ⊙ β_c )                     # [BT,BT]@[BT,V] → [BT,V]  解耦 value
kg = k_c ⊙ exp2(gk_last[None,:] − gk_c)      # [BT,K] 纯逐元素（无矩阵乘）
```

- u 没有 gk 调制 —— Value 不受 Linear Attention 的 gate 影响（注意别把 exp2(gk) 乘进去）。
- kg 不做 A_c：把每个 key 从各自时间戳对齐到 **chunk 末 token** 的时间戳
  `gk_last = gk[chunk 末有效 token, :]`，供 K5 状态递推把不同行时间衰减到同一参考点后再加。
- **kg 与 w 在同一个 tile 里顺带算**（`STORE_KG` 开关）：复用已载的 `b_k`/`b_gk`，省一次
  重复访存。

### tile 代码（= kernel 主体，注释即全部逻辑）

```
i_t, i_hg = tl.program_id(0..1)      # (chunk, head 组)；目标 HM=16，外层再套 for hh in range(HM)
b_A  = load A_inv [b, tc:tc+64, h]  [BT,BT]        # 单位下三角逆（每个 head 载一次）
b_b  = load beta [b, tc:tc+64, h]   [BT]
b_k  = load k    [b, tc:tc+64, h]   [BT,K]
b_gk = load gk   [b, tc:tc+64, h]   [BT,K]
b_kb = b_k * b_b[:, None] * exp2(b_gk)             # [BT,K] gate 调制
b_w  = tl.dot(b_A, b_kb)                           # [BT,BT]@[BT,K] → [BT,K]
store w [BT,K]
b_v  = load v  [b, tc:tc+64, h]   [BT,V]
b_vb = b_v * b_b[:, None]                          # [BT,V] 只乘 β
b_u  = tl.dot(b_A, b_vb)                           # [BT,BT]@[BT,V] → [BT,V]
store u [BT,V]
if STORE_KG:                                       # kg 与 w 同 tile，复用 b_k/b_gk
    last_idx = min(tc + BT, T) − 1                 # chunk 末有效 token
    b_gn = load gk [b, last_idx, h]  [K]
    b_kg = b_k * exp2(b_gn[None, :] − b_gk)        # [BT,K]
    store kg [BT,K]
```

> 尾 chunk 不满 64 行时 load 补 0 对齐即可 —— A_c 下三角 × 补零行仍为零，不污染有效行。

## 4. 约束与验收

- 评分 case：`B=1, T=16384, H=96, K=V=128`（NT=256，T%64==0）；输入输出 fp32。
- 正确性：三个输出 `w`/`u`/`kg` 与 `recompute_w_u_torch` 一致，逐元素差 < `1e-2`
  （当前逐位一致，max_diff=0）。
- **不许改默认 `chunk_size=64`**（破坏上游契约 = 无效解）。`A` 是给定输入（每个 chunk
  独立的单位下三角逆），不得假设各 chunk 的 `A_c` 相同而跨 chunk 复用缓存结果。

```bash
source ../env.sh                        # 容器环境（可 `source ../env.sh 4` 固定到空闲卡）
python3 test.py                         # ① 正确性：PASS + max_diff
msprof --output=./prof_k4 --application="python3 test.py --perf --repeats 7 --warmup 3" \
    && python3 test.py --report ./prof_k4        # ② 基线（msprof Task Duration 每调用均值）
```

官方基线 ≈ **3.62 ms/调用**（±10%，2026-09-07 本机 triton-ascend 3.2.1 msprof 实测口径）；
门槛 `max_diff < 1e-2`。

## 5. 设计创新点与深度解析

> 本节复盘**基线 kernel 的设计决策链**（数字为迭代环境实测，量级参考；随工具链漂移）。

- 【代数重组（跨算子协同）】`w = A@(k·β·exp2(gk))`、`u = A@(v·β)`、`kg = k·exp2(gk_last−gk)`
  把 K5 逐 chunk 递推所需的原始 k/v 改写成"每 chunk 自包含的解耦表示"：K5 从此只读
  w/u/kg 三个 [B,T,H,K]，不再需要把 A 的因果结构带进递推 —— 本 kernel 是 K5 能把
  串行递推压到每 chunk 2 个 dot 的代数前提。
- 【单 tile 直通（Route A）】`BK=K、BV=V`：无 tile 循环、无 K/V 维掩码；`T_FULL`
  constexpr 分派（T%BT==0 时连行边界检查都免）⇒ `aic_scalar` 0.359→0.094（−74%）、
  整体 1.68× —— "掩码/边界检查是标量开销主要来源"的量化证据。
- 【寄存器复用】`A_inv [BT,BT]` 与 `beta [BT]` 公共加载后留寄存器：V 维（u）与 K 维
  （w）两个 dot 共享同一份 A；kg 在 k 的访存窗口内顺带计算 —— 每个从 HBM 搬进来的
  字节被尽量多的 dot 复用。
- 【HM=16 的干净样本】CTA 24576→1536 → 6796→4514us（1.51×），且精度 **bitwise
  max_diff=0.0**：head-merge 不改变任何浮点运算顺序，是"摊薄 per-CTA 固定开销"的
  纯收益案例 —— 与 K2/K3/K6 的 HM 收益互为印证，说明瓶颈同源。
- 【负结果入库（防重复扫描）】num_warps>4 时 warp 调度开销反超收益；`tf32` 在
  triton-ascend 上等价 `ieee`（input_precision 参数无效）；512B 行对齐能提升向量
  利用率 —— 这些"试过但放弃"的结论与正收益同等重要，构成收敛论证的一部分。
