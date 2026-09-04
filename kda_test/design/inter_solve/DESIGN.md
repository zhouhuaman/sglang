# Kernel 3 设计文档: 独立 Inter-Solve Fused 算子

> 本文档是 `kda_test/design/Kernel3_InterSolve.md` 的**独立子集重构**。
> 与原始文档的差异:
> - 只保留 `B,T,H,K` 固定长度的最小闭环, 不涉及 VARLEN / safe-gate /
>   FUSE_RECOMPUTE / FUSE_DIAGONAL 等扩展路径;
> - 对角线 Akk 块由 Kernel-2（token_parallel）写好, 本 kernel 读 `Akkd` 直接做
>   前向替换 + 链式求逆;
> - 行号改为引用本目录的 `src/inter_solve_kernel.py`;
> - 增加「精度测试策略」和「性能测试思路」两节, 配套本目录测试驱动。

---

## 1. 输入输出定义

### 1.1 输入张量

| 参数名 | Shape | Dtype | 含义 |
|--------|-------|-------|------|
| `q` | `[B, T, H, K]` | fp32 | Query 张量 |
| `k` | `[B, T, H, K]` | fp32 | Key 张量 |
| `g` (gate) | `[B, T, H, K]` | fp32 | Kernel-1 的 chunk 局部 gate cumsum 输出（log2 空间） |
| `beta` | `[B, T, H]` | fp32 | Per-token per-head 的标量权重 |
| `Akkd` | `[B, T, H, BC]` | fp32 | Kernel-2 输出的对角线 Akk 块（严格下三角, j<i 同 sub-chunk 内 gated dot） |
| `scale` | 标量 | fp32 | Attention scale，通常为 `K^{-0.5}` |

编译期常量：`H`、`K`、`BT`（chunk 大小=64）、`BC`（sub-chunk 大小=16）、
`BK`（= `next_power_of_2(K)`）。

### 1.2 输出张量

| 参数名 | Shape | Dtype | 含义 |
|--------|-------|-------|------|
| `Aqk` | `[B, T, H, BT]` | fp32 | 非对角线 Aqk 块, 列 = j 在 chunk 内位置 (`j % BT`) |
| `Akk_inv` | `[B, T, H, BT]` | fp32 | 合并的下三角逆（10 个 16×16 子块）, 上三角为 0 |

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
在该 chunk 内完成 Phase 1/2/3 三阶段计算。

### 程序 ID 映射

```
i_t  = program_id(0)   chunk 索引 (0 .. NT-1)
i_hg = program_id(1)   (batch, head) 联合索引 (0 .. B*H-1)
i_b  = i_hg // H       batch 索引
i_h  = i_hg %  H       head 索引
bos  = i_b * T          batch 内起始 token 偏移
```

## 3. 计算思路

> 记号：`BT=64`、`BC=16`（一个 chunk = 4×4 个 16×16 子块）。行子块 `p` 行区间
> `r_p = i_tc*BT + 16p`，列子块 `q` 行区间 `r_q = i_tc*BT + 16q`。目标 X = Akk_inv 是
> 每 chunk 一个 **64×64 单位下三角逆**：给定 chunk 内 KK 耦合阵 `B = I + L`（`L` 严格下
> 三角，含块内与跨块），求 `X = B^{-1}`。同时算跨子块的 QK 打分（对角 Aqk 归 K2，本算子
> 只写左下 6 块）。

### 3a. 目标对象与结构：`X = B^{-1}`，`B = I + L`（4×4 块下三角）

**数学结构：**

```
B 的对角块 B_pp = I + D_p      （D_p：子块 p 内部严格下三角 KK，来自 Akkd，即 K2 对角块）
B 的左下块 B_pq = M_pq         （跨子块 KK 耦合，本 kernel 算；右上为 0）
L = B − I = strict_tril(整 chunk KK 耦合)          # 含块内下三角 + 6 个跨块
X = B^{-1} 同形状（单位下三角块阵）：X 对角块 = E_p = (I+D_p)^{-1}，左下块由逐块回代给出。
```

**分块矩阵乘法图示**（一块 chunk 的 64×64，4×4 个 16×16 子块）：

```
         列块 q →        D_p/K2 对角块 = 3a/3b 求逆输入；M_pq/Aqk_pq = 本 kernel 算
 行块 p  ┌──────────────────────────────────────┐
  0     │ I+D0  │   0   │   0   │   0   │      B 的每块：对角 = I+D_p（单位下三角），
        ├───────┼───────┼───────┼───────┤      左下 = M_pq，右上 = 0
  1     │ M_10  │ I+D1  │   0   │   0   │
        ├───────┼───────┼───────┼───────┤      X = B^{-1}：对角 = E_p，左下 X_pq
  2     │ M_20  │ M_21  │ I+D2  │   0   │      由 (B·X)=I 逐块回代（下式 6 条）
        ├───────┼───────┼───────┼───────┤
  3     │ M_30  │ M_31  │ M_32  │ I+D3  │
        └───────┴───────┴───────┴───────┘
```

### 3b. 对角块逆（Phase 2 前向替换）：`E_p = (I + D_p)^{-1}`

**数学公式：**

```
D_p[i][j] = Akkd[b, r_p+i, h, j]（只 i>j 非零）。I+D_p 单位下三角 ⇒ 逆仍单位下三角，
逐行前向代入 (I+D_p)·E_p = I（串行依赖仅 16 行，量小）。符号为 +D_p（对 −D_p 行消元）。
```

**逐块代码（4 个对角块独立并行）：**

```
A = -strict_tril(D_p)                    # 行消元初值
for i in 2..15:                          # 逐行推进
    A[i] = -D_p[i] + Σ_k A[i,k]·A[k]     # 一行行向量化
E_p = A + I
```

### 3c. 左下 6 块（Phase 1 非对角块）：`M_pq` / `Aqk_pq`

**数学公式：**

```
对行块 p、列块 q（p>q），行 token r_p+i、列 token r_q+j：
  M_pq[i][j]   = β[r_p+i] · Σ_κ k[r_p+i,κ]·k[r_q+j,κ]·exp2(g[r_p+i,κ]−g[r_q+j,κ])
  Aqk_pq[i][j] = scale    · Σ_κ q[r_p+i,κ]·k[r_q+j,κ]·exp2(g[r_p+i,κ]−g[r_q+j,κ])
Aqk_pq 写输出（对角 Aqk 归 K2 已写，块外 0）；M_pq 不写，供 3d 求逆。
β 按**行** token 广播（行块 p 的 β）—— 与上游 / 本目录 kernel 一致。
```

**逐块代码**（整块 `[16,K]@[K,16]`；指数按行块末 gate 拆两因子）：

```
Q_p,K_p,G_p = q,k,g[r_p:r_p+16]；K_q,G_q = k,g[r_q:r_q+16]；β_p = beta[r_p:r_p+16]
gni_p = G_p[15];  decR = exp2(G_p − gni_p);  decC = exp2(gni_p − G_q)   # 行/列因子
Aqk_pq = scale · (Q_p⊙decR) @ (K_q⊙decC).T
M_pq   = ((K_p⊙decR) @ (K_q⊙decC).T) * β_p[:, None]
store Aqk[行 r_p+i, 列 16q+j] = Aqk_pq[i][j]        # 只写 6 个左下块
```

### 3d. 合并求逆（Phase 3 链式回代）：`X_pq`

**数学公式（(B·X)=I 逐块展开，下标即块号）：**

```
第1层（距离 1）： X_10 = −E_1@M_10@E_0    X_21 = −E_2@M_21@E_1    X_32 = −E_3@M_32@E_2
第2层（距离 2）： X_20 = −E_2@(M_20@E_0 + M_21@X_10)
                  X_31 = −E_3@(M_31@E_1 + M_32@X_21)
第3层（距离 3）： X_30 = −E_3@(M_30@E_0 + M_31@X_10 + M_32@X_20)
```

**逐块代码 / 寄存器占用：** 对角 4 个 E_p + 左下 6 个 X_pq（10×[16,16] fp32 ≈ 10KB
寄存器），第 1 层算完才可算第 2 层、再第 3 层。写回：`Akk_inv[行 r_p+i, 列 16q+j] = X_pq[i][j]`
（X 单位下三角 ⇒ 逐块写 = 把 X 第 (16p+i) 行原样铺到输出行）。

**自检恒等式：** 拼回 64×64 后 `B·X = I`（对角=I、其余=0），块级约 fp32 精度。

### 3e. 收敛 kernel 的实现口径（融合单 kernel，数值等价、少一个依赖）

```
整 chunk 一次载入：Mkk = dot((k⊙exp2(g))·β, (k⊙exp2(−g))ᵀ)，Mqk = dot(q⊙exp2(g)·scale, ·ᵀ)
→ 两个 [64,K]@[K,64] = [64,64]；Aqk 存块严格下三角（行块>列块，对角 Aqk 归 K2）。
L = strict_tril(Mkk)（含块内下三角）；X = (I+L)^{-1} 用
    (I−L)(I+L²)(I+L⁴)(I+L⁸) 展开（L 严格下三角幂零；NP=3 级 → 6 个 dot，实测 ~1e-7）。
对角块不再读 Akkd：Mkk 的块内值在数学上等于 K2 的 Akkd（同口径），故可自给。
```

### 3f. 尾 chunk 处理

- T 非 BT 倍数时最后 chunk 的子块越界：mask（`m_t = (i_tc*BT + r) < T`）load 置 0，
  越界行不参与 dot；缓冲按 `TP=NT*BT` 补齐后无掩码全量写回（掩码处写 0）。

## 4. 关键代码对应（`src/inter_solve_kernel.py`）

| 设计逻辑 | 本目录实现 |
|---------|-----------|
| Grid 定义: (NT, B*H) | `inter_solve_triton` 中 `grid = (NT, B * H)` |
| 全局 token 索引解析 | `_inter_solve_kernel`: `i_b = i_hg // H; i_h = i_hg % H; bos = i_b * T` |
| 寄存器初始化 12 个 [BC,BC] | `b_Aqk10..b_Akk32 = tl.zeros([BC, BC], tl.float32)` |
| K 维循环加载子块 | `for i_k in range(tl.cdiv(K, BK)): tl.make_block_ptr + tl.load` |
| 非对角块 dot 累加 | `b_Aqk10 += tl.dot(b_qg1, b_kgt)` 等 |
| Aqk 存 global (带 scale) | `tl.store(p_Aqk10, (b_Aqk10 * scale).to(...))` |
| Akk 乘 beta 留寄存器 | `b_Akk10 = b_Akk10 * b_b1[:, None]` |
| 对角块从 Akkd 加载 | `b_Ai00 = tl.load(p_Akk00, boundary_check=(0,1))` |
| 前向替换逐行累加 | `for i in range(2, BC): b_a += tl.sum(b_a[:,None] * b_Ai, 0)` |
| 链式乘法第1层 | `b_Ai10 = -tl.dot(tl.dot(b_Ai11, b_Akk10), b_Ai00)` |
| 链式乘法第2层 | `b_Ai20 = -tl.dot(b_Ai22, tl.dot(b_Akk20, b_Ai00) + tl.dot(b_Akk21, b_Ai10))` |
| 链式乘法第3层 | `b_Ai30 = -tl.dot(b_Ai33, ...)` |
| 写回 10 个子块 | `tl.store(p_Akk00..p_Akk33, b_Ai00..b_Ai33)` |

## 5. torch 元算子实现（`inter_solve_torch`，性能/精度基准）

- 按 chunk 批量化：对每 chunk 的 4 个 sub-chunk 位置 `a`，提取
  `qi/ki/gi/bi = [B, NT, BC, H, K]` 切片；
- 参考点 `gni[i] = G[:,:,i*BC+BC-1,:,:].unsqueeze(2)`  —— 5D 张量切片；
- 非对角块用 `torch.einsum('bndhk,bndjk->bndhj', qi[i]*gq, bk_t)` 批量计算；
- 乘 beta_j 行广播后写回 `off[(i,j)]`；
- 对角线前向替换用 `_batch_forward_solve` 批量化（[P, BC, BC] 逐行）；
- 链式合并用 `torch.matmul` (`@`) 批量化。

它和 CPU 参考在 fp32 上逐元素一致（实测 max-diff ~1e-5），作为性能基准时
是整个 kda 算子图（einsum + matmul + 前向替换循环 + 链式合并的中间张量往返）。

## 6. 精度 & 性能对比测试策略（配套 `run.py` / `testcases.csv`）

- 每个 case 固定 seed，输入分布：`q/k = randn`，`g = randn*0.5`（log2 累积
  gate 量级较小），`beta = randn*0.1 + 1.0`（恒正、在对角线附近抖动）；
  `Akkd` 由内联 token_parallel 数学计算（逐 token 循环, 严格下三角）；
- 对每个 case 依次跑三个实现：
  1. **torch_npu 元算子** `inter_solve_torch` —— 精度基本准 + 性能基本准；
  2. **triton kernel** `inter_solve_triton` —— 被测对象；
  3. **CPU 参考** `inter_solve_ref`（逐 chunk 循环，ground truth）。
- 精度指标（两个都满足才 PASS）：
  - `max|triton - torch_npu| < 1e-2`（Aqk / Akk_inv 分开算）；
  - `max|torch_npu - ref| < 1e-2` 且 `max|triton - ref| < 1e-2`。
- 性能指标：预热 `--warmup`(默认 5) 次、各跑 `--repeats`(默认 30) 次，用
  `torch.npu.synchronize()` 包裹计时，输出每个 case 的
  **加速比 = torch_npu_time / triton_time**。
- CSV 中共 15 个 case，覆盖：完整 chunk / 尾 chunk 不满（`T=63/65/96/100/127/193/2562`）、
  单/多 head（`H=1/2/3`）、单/多 batch（`B=1/2`）、`K=32/64/128`。

## 7. 性能测试思路

- torch_npu 元算子：einsum + matmul + 前向替换串行循环 + 链式合并的算子图，
  多次 kernel 启动与中间张量读写，是性能基准的下界参考；
- triton kernel：每 chunk/head 一个 CTA，单 kernel 完成 Phase 1/2/3，
  无中间张量；K 维循环内复用同 chunk 的 k/g，寄存器内完成 dot 积；
- 每 case 预热 5 次、计时 30 次（`torch.npu.synchronize()` 包裹），报告
  `torch_ms / triton_ms / speedup`。

加速比主要来自: 单 kernel 复用与寄存器密集型计算（Phase 1 12 块 + Phase 3 10 块
[16,16] fp32 寄存器），以及避免中间张量（Aqk/Akk 块）的全局内存往返。

## 附录 A: 关键常量速查

| 常量 | 值 | 用途 |
|------|-----|------|
| `BT` | 64 | Chunk 大小 / 时间维 tile 大小 |
| `BC` | 16 | Sub-chunk 大小（NC = BT/BC = 4） |
| `BK` | `next_power_of_2(K)` | K 维 block/pad |
| `num_warps` | 1 | 每 CTA warp 数（与真实 kernel 一致） |

## 附录 B: 与真实 kernel 的差异对照

| 项 | `python/.../fla/chunk_intra.py` | 本目录独立实现 |
|----|---------------------------------------------------|----------------|
| VARLEN / `cu_seqlens` | 支持 | 只做固定长度 `bos = i_b * T` |
| `FUSE_DIAGONAL` | 对角块内联计算 | 不支持（Akkd 来自 Kernel-2） |
| `FUSE_RECOMPUTE` | 直接计算 w/u/kg | 不支持（只输出 Akk_inv） |
| `USE_SAFE_GATE` | 对角块预求逆时跳过 Phase 2 | 不支持（始终做前向替换） |
| `tl.autotune` (BK/num_warps) | 有 | 无（固定 BK=next_pow2(K), num_warps=1） |
| 无 NPU 环境 | 无法运行 | `inter_solve_ref` 退化为纯 CPU 参考 |
| 依赖 | sglang 包 | 仅 torch + torch_npu + triton |
