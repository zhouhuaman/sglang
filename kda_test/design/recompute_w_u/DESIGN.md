# Kernel 4 设计文档: 独立 Recompute W/U 算子

> 本文档是 `kda_test/design/Kernel4_RecomputeWU.md` 的**独立子集重构**。
> 与原始文档的差异:
> - 不再假设 VARLEN / `cu_seqlens` / `chunk_indices` 等扩展路径, 只保留
>   `B,T,H,K,V` 固定长度 + `STORE_KG` / `DOT_PRECISION` 的最小闭环;
> - 行号改为引用本目录的 `src/recompute_w_u_kernel.py`;
> - 增加「精度测试策略」和「性能测试思路」两节, 配套本目录测试驱动。

---

## 1. 输入输出定义

### 1.1 输入张量

| 参数名 | Shape | Dtype | 含义 |
|--------|-------|-------|------|
| `k` | `[B, T, H, K]` | fp32/bf16 | 原始 Key 张量 |
| `v` | `[B, T, H, V]` | fp32/bf16 | 原始 Value 张量 |
| `beta` | `[B, T, H]` | fp32/bf16 | Gated Delta Rule 的 beta 门控系数 |
| `A` | `[B, T, H, BT]` | fp32/bf16 | Akk_inv: chunk 内 KKT 矩阵的逆, 每 chunk `[BT, BT]` 下三角 |
| `gk` | `[B, T, H, K]` 或 None | fp32/bf16 | Key 方向的 gate cumsum（log2 空间），预乘以 exp2 使用 |
| `chunk_size` | 标量 | int | Chunk 大小 `BT=64`，等于 `A.shape[-1]` |

编译期常量：`H`、`K`、`V`、`BT`（chunk 大小=64）、`BK`（K tile=32）、
`BV`（V tile=32）、`STORE_KG`、`DOT_PRECISION`。

### 1.2 输出张量

| 参数名 | Shape | Dtype | 含义 |
|--------|-------|-------|------|
| `w` | `[B, T, H, K]` | fp32/bf16 | 解耦后的 Key 表示: `A_inv @ (k * beta * exp2(gk))` |
| `u` | `[B, T, H, V]` | fp32/bf16 | 解耦后的 Value 表示: `A_inv @ (v * beta)` |
| `kg` | `[B, T, H, K]` 或 None | fp32/bf16 | 时间对齐的 Key: `k * exp2(gk_last - gk)`（当 gk 非空时输出） |

## 2. 分核并行策略（Grid 拓扑）

```
Grid = (NT, B * H)
        ~~~   ~~~~~
         |      |
       chunk数 所有 (batch, head) 对
```

- `NT = cdiv(T, BT)`：时间维 chunk 总数；
- `B * H`：每个 (batch, head) 组合一个平面。

每个 CTA 处理一个 `(时间 chunk i_t, (b,h))`，1 个 warp（num_warps=1），
在该 chunk 内完成公共加载 + V 维循环 + K 维循环三阶段。

### 程序 ID 映射

```
i_t  = program_id(0)   chunk 索引 (0 .. NT-1)
i_bh = program_id(1)   (batch, head) 联合索引 (0 .. B*H-1)
i_b  = i_bh // H       batch 索引
i_h  = i_bh %  H       head 索引
bos  = i_b * T          batch 内起始 token 偏移
```

## 3. 计算思路

### 3.1 总体数据流

```
k [B,T,H,K]  v [B,T,H,V]  beta [B,T,H]  A=Akk_inv [B,T,H,BT]  gk [B,T,H,K]
     |            |             |                |                 |
     v            v             v                v                 v
  ┌──────────────────────────────────────────────────────────────────┐
  │ Step 1: 公共加载 (每 CTA 一次, 留寄存器)                           │
  │   beta_tile [BT]   A_inv_tile [BT, BT]                           │
  └──────────────────────────────────────────────────────────────────┘
                                |
     ┌──────────────────────────┴───────────────────────────────┐
     v                                                            v
  ┌─────────────────────────────────┐  ┌──────────────────────────────────┐
  │ Step 2: V 维循环                 │  │ Step 3: K 维循环                  │
  │ for i_v in range(cdiv(V,BV)):    │  │ for i_k in range(cdiv(K,BK)):    │
  │   v_tile = load [BT, BV]        │  │   k_tile = load [BT, BK]         │
  │   v' = v * beta                 │  │   k' = k * beta                  │
  │   u_tile = A_inv @ v'  (tl.dot)│  │   gk_tile = load [BT, BK]         │
  │   store u_tile                   │  │   k' *= exp2(gk)                  │
  └─────────────────────────────────┘  │   (if STORE_KG):                  │
                                      │     gk_last = load gk[last, :BK]  │
                                      │     kg = k * exp2(gk_last - gk)   │
                                      │     store kg_tile                 │
                                      │   w_tile = A_inv @ k'   (tl.dot)│
                                      │   store w_tile                    │
                                      └──────────────────────────────────┘
```

### 3.2 w 的计算: `w = A_inv @ (k ⊙ β ⊙ exp2(gk))`

**数学公式:**

```
w[i] = Σ_j A_inv(i, j) · k[j] · β[j] · exp2(gk[j])      （i,j = chunk 内 0..BT-1）
```

行输入先做**逐元素缩放**（`k*β`，再 `*exp2(gk)`），再乘 A_inv 解耦。

**分块矩阵乘法图示:**

```
 缩放后的输入 [BT, K]            A_inv [BT, BT]            输出 w [BT, K]
+---------------------------+   +------------------------+   +---------------------------+
| k_0·β_0·exp2(gk_0)        |   | A00  A01 ... A0,BT     |   | w_0                       |
| k_1·β_1·exp2(gk_1)        |   | A10  A11 ... A1,BT     |   | w_1                       |
| ...                       |   | ...                    | @ | ...                       |
| k_{BT-1}·β·exp2(gk_{BT-1}) |   | ABT0  ...  ABT,BT      |   | w_{BT-1}                  |
+---------------------------+   +------------------------+   +---------------------------+
             逐元素缩放                        [BT, BT]                 K dim
```

**逐 tile 分块代码（收敛 kernel：K 单 tile 直通，目标 case BK=K=128）：**

```
# grid = (NT, B·H//HM)；每 CTA 循环 HM 个头；BT=64
b_A  = load A_inv [BT, BT]                        # 单位下三角逆，每 head 一次
b_b  = load beta [BT]
b_k  = load k    [BT, K]
b_gk = load gk   [BT, K]
b_kb = b_k * b_b[:, None] * exp2(b_gk)            # [BT, K]  gate 调制
b_w  = tl.dot(b_A, b_kb, input_precision="tf32")  # [BT,BT] @ [BT,K] -> [BT,K]
store w [BT, K]
```

### 3.3 u 的计算: `u = A_inv @ (v ⊙ β)`

**数学公式:**

```
u[i] = Σ_j A_inv(i, j) · v[j] · β[j]
```

**分块矩阵乘法图示:**

```
 缩放后的输入 [BT, V]            A_inv [BT, BT]            输出 u [BT, V]
+--------------------------+   +------------------------+   +-------------------------+
| v_0·β_0                  |   | A00  A01 ... A0,BT     |   | u_0                     |
| v_1·β_1                  |   | A10  A11 ... A1,BT     |   | u_1                     |
| ...                      |   | ...                    | @ | ...                     |
| v_{BT-1}·β_{BT-1}        |   | ABT0  ...  ABT,BT      |   | u_{BT-1}                |
+--------------------------+   +------------------------+   +-------------------------+
             逐元素缩放                        [BT, BT]                 V dim
```

**逐 tile 分块代码（V 单 tile，目标 case BV=V=128）：**

```
b_v  = load v [BT, V]
b_vb = b_v * b_b[:, None]                          # [BT, V]  只乘 β
b_u  = tl.dot(b_A, b_vb, input_precision="tf32")   # [BT,BT] @ [BT,V] -> [BT,V]
store u [BT, V]
```

**注意：** u 计算没有 gk 调制，因为 Value 不受 Linear Attention 的 gate 影响。

### 3.4 kg 的计算: `kg = k ⊙ exp2(gk_last − gk)`

**数学公式:**

```
kg[i] = k[i] · exp2(gk_last − gk[i])       # gk_last = chunk 末有效 token 的 gk
```

物理含义：把 chunk 内每个 token 的 Key 从"各自时间戳"对齐到"chunk 末尾时间戳"——供
K5 状态递推时把不同行时间衰减到同一参考点后再相加。

**逐 tile 分块代码（K 单 tile，复用 3.2 已载的 b_k / b_gk）：**

```
if STORE_KG:
    last_idx = min(base + BT, T) - 1                  # chunk 最后一个有效 token
    b_gn = load gk[last_idx, :]                       # [K] chunk 末 gk
    b_kg = b_k * exp2(b_gn[None, :] - b_gk)           # [BT, K] 逐元素
    store kg [BT, K]
```

**为何 kg 搭 w 一起算而非独立循环：** 复用 3.2 已加载的 `b_k` / `b_gk`，减少重复访存。

### 3.5 为什么需要 Akk_inv（chunk 内因果依赖解耦）

#### 背景: Chunk-wise Linear Attention 的因果依赖

在 Linear Attention 中，对于 chunk 内的 token `i`，其输出为:

```
o_i = q_i^T @ sum_{j <= i} k_j v_j^T   (因果掩码)
```

其中 `sum_{j <= i} k_j v_j^T` 是一个累积的 KV 状态，在 chunk 内依赖因果顺序。

#### 引入 Akk 矩阵求逆

引入 chunk 内的 pairwise KKT 矩阵（带 gate 对齐）:

```
Akk_{m,n} = beta_n * k_m^T * k_n * exp2(gk_n - gk_last)
```

因果性由 `Akk_{m,n} = 0 for m > n` 体现（下三角矩阵）。
求逆后用 `Akk_inv` 乘以 beta 调制后的 k/v，得到**解耦的** `w` 和 `u`:

```
w = Akk_inv @ (k * beta * exp2(gk))       # Key 解耦表示
u = Akk_inv @ (v * beta)                  # Value 解耦表示
```

**核心洞察：** `w` 和 `u` 不再包含 chunk 内的因果依赖，可以直接与外积形式
`w * u^T` 参与跨 chunk 的递推计算。这就是"recompute"的含义——从原始 k/v 通过
Akk_inv 重新计算出适合跨 chunk 传播的 w/u。

### 3.6 尾 chunk / 尾 tile 的处理

- 最后一个 chunk 的行数可能不足 BT，K/V 维也可能出现最后一个 tile 不足 BK/BV；
- kernel 统一用 `tl.make_block_ptr` 的 `boundary_check` 处理: 越界位置 load 为 0，
  因此 `A @ (k * beta * exp2(gk))` 中越界行的零乘积自然为 0，不会污染有效行；
- `gk_last` 的加载用 `mask = o_k < K` 保证越界列读 0；
- `last_idx = min(i_t * BT + BT, T) - 1` 保证尾 chunk 的 gk_last 取最后一个有效 token，
  与上游 kernel 一致。

## 4. 关键代码对应（`src/recompute_w_u_kernel.py`）

| 计算步骤 | 本目录实现位置 | 代码要点 |
|----------|----------------|----------|
| Grid 定义: (NT, B*H) | `recompute_w_u_triton` 中 `grid = (NT, B * H)` | |
| CTA 索引解析 | `_recompute_w_u_kernel`: `i_t, i_bh = tl.program_id(0..1)` | `i_b = i_bh // H; i_h = i_bh % H; bos = i_b * T` |
| 公共加载 beta [BT] | `tl.make_block_ptr(beta + (bos*H + i_h), (T,), (H,), ...)` | `boundary_check=(0,)` |
| 公共加载 A_inv [BT, BT] | `tl.make_block_ptr(A + (bos*H + i_h)*BT, (T, BT), (H*BT, 1), ...)` | 一次加载, 留寄存器 |
| V 维循环头 | `for i_v in range(tl.cdiv(V, BV)):` | |
| 加载 v tile | `tl.make_block_ptr(v + (bos*H + i_h)*V, (T, V), (H*V, 1), ...)` | |
| v * beta | `b_vb = (b_v * b_b[:, None]).to(b_v.dtype)` | |
| u = A @ (v * beta) | `b_u = tl.dot(b_A, b_vb, input_precision=DOT_PRECISION)` | |
| 存出 u | `tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))` | |
| K 维循环头 | `for i_k in range(tl.cdiv(K, BK)):` | |
| 加载 k tile | `tl.make_block_ptr(k + (bos*H + i_h)*K, (T, K), (H*K, 1), ...)` | |
| k * beta | `b_kb = b_k * b_b[:, None]` | |
| 加载 gk tile | `tl.make_block_ptr(gk + (bos*H + i_h)*K, (T, K), (H*K, 1), ...)` | |
| k * beta * exp2(gk) | `b_kb = b_kb * tl.math.exp2(b_gk)` | |
| gk_last 计算 | `last_idx = tl.minimum(base + BT, T) - 1` + `tl.load(gk + ..., mask=m_k)` | `[BK]` 向量 |
| kg = k * exp2(gk_last - gk) | `b_kg = b_k * tl.math.exp2(b_gn - b_gk)` | |
| 存出 kg | `tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), boundary_check=(0, 1))` | |
| w = A @ (k * beta * exp2(gk)) | `b_w = tl.dot(b_A, b_kb.to(b_k.dtype), input_precision=DOT_PRECISION)` | |
| 存出 w | `tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))` | |

### Python 包装层

```python
# recompute_w_u_kernel.py

def recompute_w_u_triton(k, v, beta, A, gk=None, chunk_size=64, num_warps=1):
    B, T, H, K = k.shape
    V = v.shape[-1]
    BT = A.shape[-1]
    # 统一转 fp32 + 搬上 NPU (不修改调用方张量)
    k, v, beta, A = (t.to(torch.float32).to("npu") for t in (k, v, beta, A))
    has_gk = gk is not None
    gk = gk.to(...).to("npu") if has_gk else k  # dummy
    w, u = torch.empty_like(k), torch.empty_like(v)
    kg = torch.empty_like(k) if has_gk else None
    kg_ptr = kg if kg is not None else k  # dummy

    NT = _cdiv(T, BT)
    grid = (NT, B * H)
    _recompute_w_u_kernel[grid](
        k=k, kg=kg_ptr, v=v, beta=beta, w=w, u=u, A=A, gk=gk,
        T=T, H=H, K=K, V=V, BT=BT, BK=32, BV=32,
        STORE_KG=has_gk, DOT_PRECISION="tf32",
        num_warps=num_warps,
    )
    torch.npu.synchronize()
    return w, u, kg
```

### 编译期特化路径

```
                STORE_KG?       DOT_PRECISION?
                (T/F)           (tf32 / ieee)
                   │                │
    +--------------┼────────+   +───┴───+
    |               │         |   不读 gk   tf32 (上游 autotune 默认)
    |            不加载 gk        计算 kg    (triton-ascend 上等价)
    |            不输出 kg
    |               │         └───┬───┘
    |               └─────────────┘
    └────────────────────────────────┘
```

`STORE_KG` 与 `DOT_PRECISION` 共 `2 × N` 种组合, triton 编译期为每条路径
生成特化版本。本目录固定 `DOT_PRECISION="tf32"`（与上游 autotune 默认一致），
仅 `STORE_KG` 随 gk 是否非空切换。

## 5. 数据流图

### 5a. 单个 CTA 执行流程图

```
                         +------------------+
                         | CTA(i_t, i_bh)   |
                         | chunk=i_t        |
                         | bh=i_bh          |
                         +--------+---------+
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
            +-------+-------+           +-------+-------+
            | load beta[BT] |           | load A[BT,BT] |
            | from HBM     |           | from HBM      |
            +-------+-------+           +-------+-------+
                    |                           |
                    v                           v
              b_b: [BT]                   b_A: [BT,BT]
              (在寄存器)                  (在寄存器)
                    |                           |
                    +-------+-------+-----------+
                            |       |
                            |       +-------------------------------+
                            |                                       |
                            v                                       v
                +-----------+-----------+               +-----------+-----------+
                | V-dim loop             |               | K-dim loop             |
                | for i_v in 0..V/32:    |               | for i_k in 0..K/32:    |
                |                         |               |                         |
                | +-----> load v[BT,32]   |               | +-----> load k[BT,32]   |
                | |        from HBM       |               | |        from HBM       |
                | |                       |               | |                       |
                | |   v' = v * beta      |               | |   k' = k * beta       |
                | |   [BT,32] * [BT,1]   |               | |   [BT,32] * [BT,1]    |
                | |                       |               | |                       |
                | |   u = A @ v'         |               | |   load gk[BT,32]      |
                | |   tl.dot(            |               | |   from HBM             |
                | |     [BT,BT], [BT,32] |               | |                       |
                | |   ) -> [BT,32]       |               | |   k' *= exp2(gk)      |
                | |                       |               | |   [BT,32]              |
                | |   store u[BT,32]     |               | |                       |
                | |   to HBM             |               | |   if STORE_KG:         |
                | +--(next i_v)----------+               | |     gk_last[BK] =     |
                        |                                 | |       gk[last, o_k]  |
                        v                                 | |     kg = k * exp2(   |
                   (done V)                               | |       gk_last - gk)   |
                                                          | |     store kg[BT,32]    |
                                                          |                         |
                                                          |   w = A @ k'           |
                                                          |   tl.dot(              |
                                                          |     [BT,BT], [BT,32]  |
                                                          |   ) -> [BT,32]         |
                                                          |                         |
                                                          |   store w[BT,32]       |
                                                          |   to HBM               |
                                                          |                         |
                                                          +--(next i_k)------------+
                                                                       |
                                                                       v
                                                                  (done K)
                                                                       |
                                                                       v
                                                                  CTA 完成
```

### 5b. Grid 级别并行拓扑

```
Batch=2, H=3, T=256, BT=64, NT=4

Grid: (4, 6) = (NT, B*H)

    H0     H1     H2    H0     H1     H2
    B0     B0     B0    B1     B1     B1
  +------+------+------+------+------+------+
  | CTA  | CTA  | CTA  | CTA  | CTA  | CTA  |  chunk 0 (tokens 0..63)
  |(0,0) |(0,1) |(0,2) |(0,3) |(0,4) |(0,5) |
  +------+------+------+------+------+------+
  | CTA  | CTA  | CTA  | CTA  | CTA  | CTA  |  chunk 1 (tokens 64..127)
  |(1,0) |(1,1) |(1,2) |(1,3) |(1,4) |(1,5) |
  +------+------+------+------+------+------+
  | CTA  | CTA  | CTA  | CTA  | CTA  | CTA  |  chunk 2 (tokens 128..191)
  |(2,0) |(2,1) |(2,2) |(2,3) |(2,4) |(2,5) |
  +------+------+------+------+------+------+
  | CTA  | CTA  | CTA  | CTA  | CTA  | CTA  |  chunk 3 (tokens 192..255)
  |(3,0) |(3,1) |(3,2) |(3,3) |(3,4) |(3,5) |
  +------+------+------+------+------+------+

  所有 CTA 完全独立，无同步 —— 各 chunk/head 之间无数据依赖。
```

### 5c. 调用方上下文: Chunk Intra 中的位置

```
chunk_gated_delta_rule_fwd()  调用链:

  +-- Step 1: chunk_kda_scaled_dot_kkt_fwd() / token_parallel
  |      计算 Akk = beta * K_gated * K_gated^T     (chunk 内 KKT 矩阵)
  |
  +-- Step 2: chunk_kda_fwd_kernel_inter_solve_fused()
  |      下三角求解: Akk -> Akk_inv                 (求逆, 解除因果依赖)
  |
  +-- Step 3: recompute_w_u_fwd()    <-- Kernel 4 (本文档)
  |      w = Akk_inv @ (k * beta * exp2(gk))
  |      u = Akk_inv @ (v * beta)
  |      kg = k * exp2(gk_last - gk)
  |      将因果解耦后的 w, u 输出, 供后续跨 chunk 递推使用
  |
  +-- Step 4: chunk_gated_delta_rule_fwd_h()
         利用 w, u, kg 做跨 chunk 的矩阵乘累加得到 h, v_new
```

## 6. 精度 & 性能对比测试策略（配套 `run.py` / `testcases.csv`）

参考: `test_level2_kernel_precision.py::TestRecomputeWUKernel`

- 每个 case 固定 seed, 输入分布与 level2 一致:
  - `k = normalize(randn)` (L2 归一化, 模长=1);
  - `v = randn * 0.1` (小量级, 避免递推放大);
  - `raw_gate = randn*0.5 - 2.0` (log2 空间, 对应 gate cumsum);
  - `beta = sigmoid(rand)` (对角缩放, 恒正, 0~1);
  - `A = Akk_inv 近似` (单位下三角 + alpha=0.1 随机扰动, 保持自包含);
  - `scale = K^{-0.5}`.
- 对每个 case 依次跑**三个实现**:
  1. **torch_npu 元算子** `recompute_w_u_torch` —— 精度基本准 + 性能基本准;
  2. **triton kernel** `recompute_w_u_triton` —— 被测对象;
  3. **CPU 参考** `recompute_w_u_ref` (逐 chunk 循环, ground truth).
- 精度指标 (三个都满足才 PASS):
  - `max|triton - torch_npu| < 1e-2` (w / u / kg 分开算);
  - `max|torch_npu - ref| < 1e-2` 且 `max|triton - ref| < 1e-2`.
- 性能指标: 预热 `--warmup`(默认 5) 次、各跑 `--repeats`(默认 30) 次, 用
  `torch.npu.synchronize()` 包裹计时, 输出每个 case 的
  **加速比 = torch_npu_time / triton_time**.
- CSV 中共 15 个 case, 覆盖: 完整 chunk / 尾 chunk 不满 (`T=63/65/96/100/127/193/2562`)、
  单/多 head (`H=1/2/3`)、单/多 batch (`B=1/2`)、`K=32/64/128`
  (同时覆盖 `BK=32` 整数倍与非整数倍)。

## 7. 性能测试思路

- torch_npu 元算子: `permute + matmul + reshape + exp2 + elementwise mul` 的算子图,
  多次 kernel 启动 + 中间张量读写, 是性能基准的下界参考;
- triton kernel: 每 chunk/head 一个 CTA, 单 kernel 完成 V 循环 + K 循环 + (kg),
  无中间张量; A_inv 与 beta 复用 (留寄存器), 访存主要来自 k/v/gk 加载;
- 每 case 预热 5 次、计时 30 次 (`torch.npu.synchronize()` 包裹), 报告
  `torch_ms / triton_ms / speedup`.

加速比主要来自: 单 kernel 复用 A_inv (寄存器密集型), 以及避免中间张量
(`v*beta`、`k*beta`、`k*beta*exp2(gk)` 的全局内存往返)。由于 kernel 是访存密集型
(每个 CTA 只有少量 dot 计算, 主要开销在 HBM 读写), 理论加速比接近
「算子图启动 / 中间读写」的节省比例。

## 附录 A: 关键常量速查

| 常量 | 值 | 用途 |
|------|-----|------|
| `BT` | 64 | Chunk 大小 / 时间维 tile 大小 |
| `BK` | 32 | K 维 tile 大小 |
| `BV` | 32 | V 维 tile 大小 |
| `num_warps` | 1 | 每 CTA warp 数 (与上游 autotune 默认一致) |
| `DOT_PRECISION` | `"tf32"` | `tl.dot` 精度 (triton-ascend 上等价于 ieee) |

## 附录 B: 与真实 kernel 的差异对照

| 项 | `python/.../fla/kda.py` | 本目录独立实现 |
|----|---------------------------------------------------|----------------|
| VARLEN / `cu_seqlens` / `chunk_indices` | 支持 | 只做固定长度 `bos = i_b * T` |
| `tl.autotune` (BK/BV/num_warps) | 有 (固定 BK=32/BV=32/num_warps=1) | 无 (固定常量) |
| `STORE_KG` | 由 `gk is not None` 决定 | 一致 |
| `DOT_PRECISION` | `"tf32"` | `"tf32"` (一致) |
| 无 NPU 环境 | 无法运行 | `recompute_w_u_ref` 退化为纯 CPU 参考 |
| 依赖 | sglang 包 | 仅 torch + torch_npu + triton |
