# Kernel 6 设计文档: 独立 GLA Output 算子

> 本文档是 `kda_test/design/Kernel6_GLAOutput.md` 的**独立子集重构**。
> 与原始文档的差异:
> - 只保留固定长度 `B,T,H,K,V` 的最小闭环, 去掉 VARLEN / `cu_seqlens` /
  `chunk_indices` 扩展路径;
> - 行号改为引用本目录的 `src/gla_output_kernel.py`;
> - 增加「精度测试策略」和「性能测试思路」两节, 配套本目录测试驱动。

---

## 1. 输入输出定义

### 1.1 输入张量

| 参数名 | Shape | Dtype | 含义 |
|--------|-------|-------|------|
| `q` | `[B, T, H, K]` | bf16/fp16 | Query 向量 |
| `v_new` | `[B, T, H, V]` | bf16/fp16 | Value 向量（Kernel 5 Delta Rule 修正后） |
| `g` | `[B, T, H, K]` | fp32 | 累积 gate（Kernel 1 chunk-local cumsum 输出，log2 空间） |
| `h` | `[B, NT, H, V, K]` | fp32 | 每 chunk 开始时的压缩状态快照（Kernel 5 输出） |
| `A` (Aqk) | `[B, T, H, BT]` | bf16 | chunk 内因果注意力矩阵 |
| `scale` | 标量 | float | 注意力缩放 `1/sqrt(K)` |
| `chunk_size` (BT) | 标量 | int | Chunk 大小（默认 64） |

### 1.2 输出张量

| 参数名 | Shape | Dtype | 含义 |
|--------|-------|-------|------|
| `o` | `[B, T, H, V]` | bf16/fp16 | 最终注意力输出 |

### 1.3 编译期常量

| 符号 | 典型值 | 含义 |
|------|--------|------|
| `BT` | 64 | Chunk 大小 |
| `BK` | 32 | K 维 tile 大小 (autotune) |
| `BV` | 32 | V 维 tile 大小 (autotune) |
| `NT` | `cdiv(T, BT)` | Chunk 总数 |

---

## 2. 分核并行策略

### 2.1 Grid 拓扑

```
Grid = (cdiv(V, BV), NT, B * H)
        ~~~~~~~~~~~  ~~  ~~~~~
           |          |      |
        V 维 tile   chunk数  所有 (batch, head) 对
```

- `cdiv(V, BV)`: V 按 `BV=32` 切分得到的 V-tile 数;
- `NT = cdiv(T, BT)`: 时间维 chunk 总数;
- `B * H`: 每个 (batch, head) 组合一个平面。

每个 CTA 处理一个 `(V-tile i_v, chunk i_t, (b, h))` 交集，输出 `[BT, BV]` 子块。

### 2.2 程序 ID 映射

```
i_v  = program_id(0)   V 维 tile 索引 (0 .. cdiv(V,BV)-1)
i_t  = program_id(1)   chunk 索引 (0 .. NT-1)
i_bh = program_id(2)   (batch, head) 联合索引 (0 .. B*H-1)
i_b  = i_bh // H       batch 索引
i_h  = i_bh %  H       head 索引
```

### 2.3 为什么 K 维用 sequential loop

K 维度**没有被 Grid 并行化** —— 它使用 sequential loop:

```
for i_k in range(cdiv(K, BK)):
    # 加载 q[BT, BK], g[BT, BK], h[BV, BK]
    # 累加到 b_o: b_o += dot(q_gated[BT, BK], h_t[BV, BK])
```

因为 K 维度被拆成 BK=32 的小块后，每次只需 `[BT,BK] * [BV,BK]^T` 的矩阵乘，
计算量相对轻量，留作 sequential loop 不显著增加总延迟。

---

## 3. 计算思路

Kernel 6 是 KDA 的收尾算子: 每个 token 的输出 = **跨块**（用自己的 query 读"本 chunk
之前所有 chunk"的压缩状态）+ **块内**（对同 chunk 内 ≤ 自己的 token 做精确因果注意力）。
两路各自只是一条矩阵乘，**没有跨 chunk 串行依赖**，(chunk, head) 完全并行。

> 口径注: §2/§4/§7 与附录中的 `BK=32/BV=32/num_warps=1` 是早期配置; 本目录
> `src/gla_output_kernel.py`（与 interview 版同一收敛实现）已收敛到
> **`BK=min(K,128)`、`BV=128`、HM=16 头合并**（grid 第三维 `B·(H//HM)`，与 head 无关的
> 因果 mask / 边界 mask / 偏移向量全部提升到头循环外——标量寻址削减，见
> OPTIMIZATION_LOG.md 第五轮，目标 case 6.94→4.75ms）。目标 case K=V=128 →
> `cdiv(K,BK)=1`，跨块即单条 `[64,128]@[128,128]` 大 dot。数学完全等价。

总式（每个 CTA 把两路累进同一个 fp32 累加器再写回）:

```
对 token t（chunk c、块内行 i）:
  o[t] = ( q[t]·exp2(g[t])·scale ) @ h_cᵀ      # ① 跨块: 读 chunk 之前的压缩历史
       + tril(A_c) 行 i @ v_new                # ② 块内: 同 chunk 内 j≤i 的精确注意力
h_c = h[b,c,h] 是 K5 快照，只含 chunk c 之前的信息（天然 causal，无需掩码）;
A_c = Aqk[b,tc:tc+BT,h] 是 K2/K3 块内打分（已带 scale），只需把 j>i 清 0。
```

### 3.1 跨块部分: `o_cross = (q ⊙ exp2(g) · scale) @ h_cᵀ`

**数学公式：**

```
o_cross[i, v] = Σ_κ  q[i,κ] · exp2(g[i,κ]) · scale · h_c[v,κ]
query 先按自己 token 的逐通道 gate 衰减、再乘 scale（≈ 自己"此刻"的强度），去点乘状态
第 v 行 —— 读到"所有更早 chunk"的信息。h 实现 O(T_hist) → O(K·V) 的信息压缩。
```

gate 语义: `g` 越负（累计衰减越大）→ `exp2(g)` 越小 → 该通道历史信息衰减越多; `g≈0`
→ `exp2(g)≈1` → 完整保留。

**分块矩阵乘法图示：**

```
 qg = q·exp2(g)·scale [BT, K]         h_cᵀ [K, V]          o_cross [BT, V]
+---------------------+   +--------------------+   +----------------------+
| qg_0: ——— K 通道 ———  |   | 转置: 第 v 列 =      |   | o_cross_0            |
| qg_1                 |   | state 第 v 行;       | @ | o_cross_1            |
| ...                 | @ | 第 κ 行 = 第 κ 通道  | = | ...                  |
| qg_{BT-1}            |   |                    |   | o_cross_{BT-1}       |
+---------------------+   +--------------------+   +----------------------+
  行 i = 自己的 query（已 gate/scale）   记忆只读（chunk 之前的压缩历史）   [BT,V]
```

**逐 tile 分块代码（BK=min(K,128)，目标 case 无 K 循环；每 CTA 循环 HM 个头）:**

```
for hh in tl.range(HM):                        # head-merged 循环; b_o 每 head 清零
    b_o  = tl.zeros([BT, BV], fp32)
    b_q  = load q  [i_t*BT:(i_t+1)*BT, :]  ;  b_q = b_q * scale      # [BT,BK]
    b_g  = load g  [i_t*BT:(i_t+1)*BT, :]                             # [BT,BK]
    b_qg = b_q * tl.math.exp2(b_g)                                     # [BT,BK]
    b_h  = load h  [i_tg, i_v*BV:(i_v+1)*BV, :]                        # [BV,BK]
    b_o += tl.dot(b_qg, tl.trans(b_h))        # [BT,BK]@[K,BV] → [BT,BV]
```

### 3.2 块内部分: `o_intra = tril(A_c) @ v_new`

**数学公式：**

```
o_intra[i, v] = Σ_{j≤i} A_c[i, j] · v_new[j, v]      # 只加 j≤i（本 chunk 内的历史/自己）
掩码 m_s[i,j] = (i ≥ j)：上三角 j>i 置 0。
```

**分块矩阵乘法图示：**

```
 tril(A_c) [BT, BT]          v_new_c [BT, V]         o_intra [BT, V]
+------------------------+   +--------------------+   +----------------------+
| ▣ ▣                    |   | v_0: ——— V 通道 ———  |   | o_intra_0（行 i 只累加 |
| ▣ ▣ ▣                  |   | v_1                 | @ |   j≤i 的 v_new）     |
| ... 下三角含对角 ▣      |   | ...                 | = | ...                  |
| ▣ ▣ ... ▣              |   | v_{BT-1}            |   | o_intra_{BT-1}      |
+------------------------+   +--------------------+   +----------------------+
  行 i 只保留列 j≤i（因果）        块内每个 token 的修正 value          [BT,V]
```

**逐 tile 分块代码：**

```
    b_A = load A [i_t*BT:(i_t+1)*BT, :BT]                     # [BT,BT] 打分
    b_A = tl.where(m_s, b_A, 0.0)                             # 因果 mask 提前乘进 A
    b_v = load v_new [i_t*BT:(i_t+1)*BT, i_v*BV:(i_v+1)*BV]   # [BT,BV]
    b_o += tl.dot(b_A, b_v)                                   # [BT,BT]@[BT,BV] → [BT,BV]
```

### 3.3 相加与写回

```
store o [i_t*BT:(i_t+1)*BT, i_v*BV:(i_v+1)*BV] = b_o          # 两路已在累加器相加
```

fp32 累加器 `b_o` 先累 ① 跨块、再累 ② 块内，最后一次性 cast 到输出 dtype（目标 case
fp32）写回——不产生 `qg` / `o_cross` 中间张量落盘。

### 3.4 尾 chunk / 尾 V-tile 处理

最后一个 chunk 的行数可能不足 BT、最后一个 V-tile 可能不足 BV。kernel 用行/列
`mask`（手动指针算术版）处理: 越界元素 load 为 0；`tl.where(m_s, b_A, 0.0)` 施加因果
mask 后，越界行的 `b_A` 也被清零，因此不会污染累加器（参考实现同样补 0 再裁）。

---

## 4. 关键代码对应 (`src/gla_output_kernel.py`)

- 内核: `chunk_gla_fwd_kernel_o` (`@triton.jit`)
  - program ID 与索引计算: `i_v, i_t, i_bh = tl.program_id(0..2)`;
  - 因果 mask: `m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]`;
  - K 维 sequential loop: 加载 `[BT, BK]` q/g + `[BV, BK]` h，
    `b_o += tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype))`;
  - 块内: 加载 `[BT, BT]` Aqk + `[BT, BV]` v_new，
    `b_o += tl.dot(b_A_masked, b_v)`;
  - grid: `(cdiv(V, BV), cdiv(T, BT), B * H)`.
- 驱动: `gla_output_kernel(...)` (NPU: triton; 无 NPU 时退化为 `gla_output_ref`).
- CPU 参考: `gla_output_ref(...)` (纯 torch, 逐 chunk 循环).
- torch_npu 基准: `gla_output_torch(...)` (批量 matmul + tril + exp2).

### Autotune 配置

```python
@triton.autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32]
        for BV in [32]
        for num_warps in [1]
        for num_stages in [1]
    ],
    key=["BT"],
)
```

与上游 `kda.py` 的 autotune 配置完全一致 (BK=32, BV=32, num_warps=1, num_stages=1),
key 为 `["BT"]` (固定长度模式下 IS_VARLEN 恒为 False，已省略)。

---

## 5. 数据流图

```
                       输入
                        |
       ┌────────────────┼────────────────────────────────┐
       |                |                                |
       v                v                                v
   q [B,T,H,K]    v_new [B,T,H,V]              g [B,T,H,K]  (log2 空间)
       |                |                                |
       |                |    h [B,NT,H,V,K]    Aqk [B,T,H,BT]
       |                |       |                    |
       v                v       v                    v
┌──────────────────────────────────────────────────────────────┐
│ Grid: (cdiv(V,BV), NT, B * H)                                 │
│                                                               │
│ CTA(i_v, i_t, i_bh):                                          │
│   - V 维: tile i_v, BV=32                                    │
│   - 时间维: chunk i_t, BT=64                                 │
│   - batch: i_b = i_bh // H                                   │
│   - head:  i_h = i_bh %  H                                   │
│                                                               │
│  ┌─── 跨块路径 (K-loop) ──────┐  ┌─── 块内路径 ────────────┐ │
│  │ for i_k in K/BK:           │  │ b_v = load(v[chunk, V]) │ │
│  │   b_qg = q*scale*exp2(g)   │  │ b_A = load(Aqk[chunk])  │ │
│  │        [BT,BK]              │  │ b_A = where(m_s, b_A, 0)│ │
│  │   b_h = load(h[V-tile,K])  │  │                          │ │
│  │        [BV,BK]              │  │ o_intra = dot(A_masked,  │ │
│  │   b_o += dot(b_qg, b_h^T)  │  │           b_v)  [BT,BV]  │ │
│  │        [BT,BV]              │  │                          │ │
│  └──────────┬─────────────────┘  └──────────┬───────────────┘ │
│             |                                |                 │
│             └────────── b_o (fp32 累加器) ───┘                 │
│                            |                                   │
│                            v                                   │
│                     store o[chunk, V-tile]                     │
│                     (cast → bf16/fp16)                         │
└──────────────────────────────────────────────────────────────┘
                        |
                        v
                  output [B, T, H, V] (bf16/fp16)
```

### 跨块路径内存访问

```
q, g 内存布局: [B, T, H, K], stride = (T*H*K, H*K, K, 1)
  block_ptr:
    base    = q + (bos * H + i_h) * K    ← 定位到 batch 起始 + head
    shape   = (T, K)
    strides = (H * K, 1)
    offsets = (i_t * BT, i_k * BK)
    block   = (BT, BK)
    order   = (1, 0)                     ← K 维连续

h 内存布局: [B*NT, H, V, K], stride = (H*V*K, V*K, K, 1)
  block_ptr:
    base    = h + (i_tg * H + i_h) * V * K   ← i_tg = i_b * NT + i_t
    shape   = (V, K)
    strides = (K, 1)
    offsets = (i_v * BV, i_k * BK)
    block   = (BV, BK)
    order   = (1, 0)                        ← K 维连续
```

---

## 6. 精度 & 性能对比测试策略 (配套 `run.py` / `testcases.csv`)

参考: `kda_test/test_level2_kernel_precision.py::TestGLAOutputKernel`

- 每个 case 固定 seed, 输入分布与 level2 一致
  (`raw_gate = randn*0.5-2.0`, `A_log*0.1`, `dt_bias*0.1`,
  `q/k` 为 L2-normalized, `v = randn*0.1`, `beta = sigmoid(rand)`,
  `initial_state = randn*0.05`);
- Kernel 6 的输入 `g / v_new / Aqk / h` 由 NPU 上的 K1-K5 流水线产生,
  与 level2 完全一致:
  1. `kda_gate_chunk_cumsum` → `g_cumsum` (K1)
  2. `chunk_kda_fwd_intra` → `w, u, kg, Aqk` (K2-K4)
  3. `chunk_gated_delta_rule_fwd_h` → `h, v_new` (K5)
- 对每个 case 依次跑**两个对比方**:
  1. **torch_npu 元算子** `gla_output_torch` —— 用 torch_npu 现成算子组合
     (`exp2`→`mul`→批量 `matmul`→`tril`→`mul`→批量 `matmul`→`add`) 完成同样计算,
     作为**精度基本准**和**性能基本准**;
  2. **triton kernel** `gla_output_kernel` —— 本目录实现的单 kernel 版本.
- 精度指标 (三者都满足才 PASS):
  - `max|triton - torch_npu| < 1e-2`;
  - `max|torch_npu - CPU 参考| < 1e-2` 且 `max|triton - CPU 参考| < 1e-2`
    (CPU 参考 `gla_output_ref` 为逐 chunk 循环的 ground truth).
- 性能指标: 预热 `--warmup` (默认 5) 次、各跑 `--repeats` (默认 30) 次,
  用 `torch.npu.synchronize()` 包裹计时, 输出每个 case 的
  **加速比 = torch_npu_time / triton_time**.
- CSV 中共 10 个 case, 覆盖: 完整 chunk / 尾 chunk 不满
  (`T=63/65/96/100/127`)、单/多 head (`H=1/2/3`)、单 token (`T=1/2`).
  K=V=64 固定 (上游 `chunk_delta_h` kernel 要求 K=64).

> 预期: 10/10 PASS; maxdiff 应在 fp32 累积噪声级 (≤ ~1e-3)。

---

## 7. 性能测试思路 (配套 `run.py`)

Kernel 6 是计算 + 内存混合型 kernel:
- 跨块路径: `[BT,BK] @ [BK,BV]` × (K/BK) 次矩阵乘 (计算密集);
- 块内路径: `[BT,BT] @ [BT,BV]` 一次矩阵乘 (计算密集);
- 总数据移动: 读 `q[B,T,H,K]` + `v_new[B,T,H,V]` + `g[B,T,H,K]`
  + `Aqk[B,T,H,BT]` + `h[B,NT,H,V,K]`，写 `o[B,T,H,V]` (内存密集).

性能对比的两个对象:
- **torch_npu 元算子**: `exp2 + mul + matmul + tril + mul + matmul + add`
  的算子图, 多次 kernel 启动 + 中间张量读写 (尤其 `qg[B,NT,BT,H,K]`
  与 `o_cross[B,NT,BT,H,V]` 两个大中间张量), 是性能基准的下界参考;
- **triton kernel**: 单 kernel 完成跨块 + 块内 + 相加, 避免中间张量;
  每 case 预热 5 次、计时 30 次, 报告 `torch_ms / triton_ms / speedup`.

加速比主要来自: 单 kernel 复用 (中间张量 `qg` 和 `o_cross` 不落盘) +
tile 选择 (`BK=BV=32` 匹配上游). 实测加速比取决于 case 大小, 大 T
case 中间张量更大, 加速比应更高。

---

## 附录 A: 关键常量速查

| 常量 | 值 | 用途 |
|------|-----|------|
| `BT` | 64 | Chunk 大小 / 时间维 tile 大小 |
| `BK` | 32 | K 维 tile 大小 |
| `BV` | 32 | V 维 tile 大小 |
| `RCP_LN2` | 1.4426950216293335 | ln(2) 倒数, ln→log2 转换 (K1 用) |
| `num_warps` | 1 | 每 CTA warp 数 (与真实 kernel 一致) |

---

## 附录 B: 与真实 kernel 的差异对照

| 项 | `python/.../fla/kda.py` | 本目录独立实现 |
|----|--------------------------|----------------|
| VARLEN / `cu_seqlens` | 支持 | 只做固定长度 |
| `chunk_indices` 预计算 | 有 | 无 |
| `autotune` key | `["BT", "IS_VARLEN"]` | `["BT"]` (IS_VARLEN 恒 False) |
| `BK`/`BV` 候选 | `[32]` / `[32]` | `[32]` / `[32]` (一致) |
| 无 NPU 环境 | 无法运行 | `gla_output_ref` 退化为纯 CPU 参考 |
| 依赖 | sglang 包 + 上游 K1-K5 | 仅 torch + triton + sglang K1-K5 |
