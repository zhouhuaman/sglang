# 问题 K3 · inter_solve

> 目录文件：`inter_solve_kernel.py` 里有 torch 参考 `inter_solve_torch` 与待改写/待优化的
> triton kernel `inter_solve_triton`（当前为多轮收敛版）。你只改 kernel；`test.py` 负责
> 比对与计时。

## 0. 算子描述

一个 chunk = 64 token = 4 个 16-token 子块（行块 `p`、列块 `q`，p,q=0..3）。只有"后面的
行块 `p` × 前面的列块 `q`（q≤p）"有意义。本算子输出两块：`Akk_inv` = 整 chunk 块下三角
单位阵（对角块 + 6 个 KK 耦合块 `M_pq`）的逆，`Aqk` = 其中 6 个跨子块（q<p）的 QK
衰减打分。每个 (batch, chunk) 独立。收敛实现不再逐子块读写 `Akkd`（§3 用整 chunk
`Mkk` 一把算全，其对角 16 块数值恰等于 `Akkd`）。

## 1. 输入

| 张量 | shape | 含义 |
|---|---|---|
| `q` | [B, T, H, K] | 每个 token 的 query 向量 |
| `k` | [B, T, H, K] | 每个 token 的 key 向量 |
| `g` | [B, T, H, K] | 每个 token 逐通道的 gate（log2 空间） |
| `beta` | [B, T, H] | 每个 token 一个标量门 |
| `Akkd` | [B, T, H, 16] | 子块内部的 KK 对角块（K2 输出；§3 里它的数值 = `Mkk` 的对角 16×16 块，故收敛 kernel 不再读它） |
| `scale` | float | 常量 `1/sqrt(K)` |

维度：`B`=batch；`T`=token 序号；`H`=head；`K`=特征通道。切块：`BT=64`、`BC=16`。

## 2. 输出

| 张量 | shape | 含义 |
|---|---|---|
| `Aqk` | [B, T, H, 64] | 6 个左下块 `Aqk_pq`（p>q）；其余 0 |
| `Akk_inv` | [B, T, H, 64] | 64×64 单位下三角矩阵 `X`，逐 token 行存放 |

## 3. 计算（先划 tile → 单 tile 内全部计算）

> 记号：`BT=64`；一个 chunk = 4 个 16-token 子块（行块 `p`、列块 `q`，p,q=0..3），行 token
> 属于行块 `r//16`。本算子把整 chunk 的**块下三角单位阵求逆**算成 `Akk_inv`，同时把跨子块
> （p>q）的 6 个 QK 打分 `Aqk_pq` 写出（对角 4 块的 Aqk 归 K2 负责，本算子不写）。每个
> (batch, chunk, head) 独立。

### 划 tile：每 CTA = 一个 (chunk, HM 个头) 的整块

收敛 kernel 一个 CTA 处理**整 chunk 的 64 行 × 全部 K 通道**、循环 `HM` 个头（目标
HM=16）。输入只需 q/k/g/β（不另读 Akkd）：

```
grid = (cdiv(T,BT), B·(H//HM))     # → (NT, B·H//16)；tile = (chunk, HM 个头)
整 chunk 载入后 2 次 [BT,K]@[K,BT] 大 dot 得 Mkk/Mqk，再做块掩码 + 幂级求逆
```

### 单 tile 公式（tile = (b,h) 的一个整 chunk）

先把整 chunk 的 QK / KK 衰减打分一次算全（`exp2(g[i]−g[j]) = exp2(g[i])·exp2(−g[j])`
拆行/列因子）：

```
Mkk = ( k⊙exp2(g)⊙β ) @ ( k⊙exp2(−g) )ᵀ     # KK 口径 [BT,BT]：对角 16 块==Akkd、
                                            #   p>q 块 == 早先的 M_pq
Mqk = ( q⊙exp2(g)·scale ) @ ( k⊙exp2(−g) )ᵀ  # QK 口径 [BT,BT]：p>q 块 == Aqk_pq
L  = strict_tril(Mkk)                        # 整 64×64 严格下三角（含块内）
X  ≈ (I−L)(I+L²)(I+L⁴)(I+L⁸)                # (I+L)⁻¹ 的截断幂级（L 严格下三角幂零）
Aqk   ← Mqk 掩到"行块 p>列块 q"（6 块）       # 对角 4 块 Aqk 由 K2 写，此处置 0
Akk_inv ← X                                 # 整 chunk 单位下三角逆，逐行铺 [B,T,H,64]
```

- 对角块不再另读 Akkd 求逆：`Mkk` 的对角 16×16 块内部值与 `Akkd[b, r_p+i, h, j]` 相同，
  故 L 已把块内与跨块耦合一并编码。
- `X = (I−L)(I+L²)(I+L⁴)(I+L⁸)` 是 `B = I+L`（对角块 I+D_p、左下块 M_pq）求逆的幂级
  展开（I − L + L² − … 截断；NP=3 级 = 6 个 dot，实测精度 ~1e-7）。

### tile 代码（= kernel 主体，注释即全部逻辑）

```
i_tc, i_hg = tl.program_id(0..1)      # (chunk, head 组)；越界 chunk 直接 return
# （目标 HM=16：外层再套 for hh in tl.range(HM)；以下是一个 head 的 body）
载整 chunk: q/k/g [BT,K]、β [BT]（行越界补 0）
b_eg  = exp2(g) ;  b_Ke = k * exp2(−g)
b_Mkk = tl.dot(k * b_eg * β[:,None], tl.trans(b_Ke))   # KK [BT,BT]
b_Mqk = tl.dot(q * b_eg * scale,       tl.trans(b_Ke))  # QK [BT,BT]
tl.store(Aqk, tl.where(行块 p > 列块 q, b_Mqk, 0.0))     # 只写 p>q 的 6 块
b_L   = tl.where(r[:,None] > c[None,:], b_Mkk, 0.0)     # 整块严格下三角 L
b_inv = I − L ;  b_pow = b_L
b_pow = tl.dot(b_pow, b_pow)      # L²
b_inv = tl.dot(b_inv, I + b_pow)  # × (I+L²)
b_pow = tl.dot(b_pow, b_pow)      # L⁴
b_inv = tl.dot(b_inv, I + b_pow)  # × (I+L⁴)        (NP=2)
b_pow = tl.dot(b_pow, b_pow)      # L⁸
b_inv = tl.dot(b_inv, I + b_pow)  # × (I+L⁸)        (NP=3)
tl.store(Akk_inv, b_inv)          # 整 chunk 逆，逐 token 行铺 [B,T,H,64]
```

> 这是与"读 Akkd、逐 16 子块前向代入求 E_p、再做 6 块回代"**数值等价的融合口径**：少一层
> 依赖、把零碎小块并成整 chunk 两次大 dot + 截断幂级。自检恒等式：把 X 逐行拼回 64×64，
> 应满足 `(I+L)·X = I`（块级约 fp32 精度）。尾 chunk 不满 64 行：越界行补 0，L 严格下三角
> × 补零行仍为 0，不污染有效行。

## 4. 约束与验收

- 评分 case：`B=1, T=16384, H=96, K=128`（T%64==0）；输入输出 fp32。
- 正确性：与 `inter_solve_torch` 逐元素差 < `1e-2`（当前 ~3e-7）。
- **不许改默认 `chunk_size=64 / sub_chunk_size=16`**（破坏上游契约 = 无效解）。
- 尾 chunk 不满 64 时参考会补 0 再裁，真实 token 值不受影响。

```bash
source ../env.sh                        # 容器环境（可 `source ../env.sh 4` 固定到空闲卡）
python3 test.py                         # ① 正确性：PASS + max_diff
msprof --output=./prof_k3 --application="python3 test.py --perf --repeats 7 --warmup 3" \
    && python3 test.py --report ./prof_k3        # ② 基线（msprof Task Duration 每调用均值）
```

官方基线 ≈ **14.4–14.9 ms/调用**（±10%）；门槛 `max_diff < 1e-2`。
