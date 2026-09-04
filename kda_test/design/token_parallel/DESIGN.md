# Kernel 2 设计文档: 独立 Token-Parallel 对角线 Aqk/Akk 算子

> 本文档是 `kda_test/design/Kernel2_TokenParallel.md` 的**独立子集重构**。
> 与原始文档的差异:
> - 只保留 `B,T,H,K` 固定长度的最小闭环, 不涉及 VARLEN / `cu_seqlens`;
> - 行号改为引用本目录的 `src/token_parallel_kernel.py`;
> - 增加「精度测试策略」和「性能测试思路」两节, 配套本目录测试驱动;
> - 增加「torch 元算子实现的对称性」一节, 说明 `token_parallel_torch` 与
>   CPU 参考 / triton 数学等价。

---

## 1. 输入输出定义

### 1.1 输入张量

| 参数名 | Shape | Dtype | 含义 |
|--------|-------|-------|------|
| `q` | `[B, T, H, K]` | fp32 | Query 张量 |
| `k` | `[B, T, H, K]` | fp32 | Key 张量 |
| `g` (gate) | `[B, T, H, K]` | fp32 | Kernel-1 的 chunk 局部 gate cumsum 输出（log2 空间），
| `beta` | `[B, T, H]` | fp32 | Per-token per-head 的标量权重 |
| `scale` | 标量 | fp32 | Attention scale，通常为 `K^{-0.5}` |

编译期常量：`H`、`K`、`BT`（chunk 大小=64）、`BC`（sub-chunk 大小=16）、
`BK`（= `next_power_of_2(K)`）、`BH`（每 CTA head 数=1）。

### 1.2 输出张量

| 参数名 | Shape | Dtype | 含义 |
|--------|-------|-------|------|
| `Aqk` | `[B, T, H, BT]` | fp32 | 对角线区域 Aqk 结果，列 = j 在 chunk 内位置 (`j % BT`) |
| `Akk` | `[B, T, H, BC]` | fp32 | 对角线区域 Akk 结果，列 = sub-chunk 内相对偏移 (`j - i_ts`) |

## 2. 分核并行策略（Grid 拓扑）

```
Grid = (B * T, cdiv(H, BH))    其中 BH = 1
        ~~~~            ~~~~
          |               |
       token 维      head 组维
```

- `program_id(0)` = `i_tg`：全局 token 索引（`0 .. B*T-1`）；
- `program_id(1)` = `i_hg`：head 组索引（每 CTA 处理 1 个 head）。

总 CTA 数 = `B * T * H`。每个 (batch, token, head) 三元组一个 CTA——最大粒度
并行，适合小 batch / 变长场景，避免在 padding token 上浪费计算。

### 程序 ID 映射（固定长度）

```
bos = (i_tg // T) * T    -- batch 内起始 token 偏移
i_t = i_tg %  T          -- batch 内局部 token 索引
i_c = i_t // BT          -- chunk 索引
i_s = (i_t % BT) // BC   -- sub-chunk 索引 (0..3)
i_ts = i_c*BT + i_s*BC   -- sub-chunk 起始 token(全局)
```

## 3. 计算思路

> 口径注：本目录 `src/token_parallel_kernel.py` 已收敛到 **Route C**（每 CTA 一次载入整
> chunk、两次 `[BT,K]@[K,BT]` 大 `tl.dot`、再按"对角 16×16 块 ∧ 块内因果"掩码；HM 头合并）。
> §2 的 per-token grid（`(B*T,H)`）是早期 Route 的并行口径；**数学与 Route C 完全等价**，
> 只是把同一 chunk 的 4 个窗口拼成一次大矩阵乘。以下计算逻辑按收敛 kernel 口径描述。

### 3a. Aqk 的计算（query×gated-key 窗口打分）：`Aqk[t, j%BT] = scale·⟨q[t], k[j]⊙exp2(g[t]−g[j])⟩`

**数学公式：**

```
对 chunk 内行 token i 与列 token j（同一窗口 s=(行//16)*16，且 j ≤ i）：
  Aqk[i, j] = scale · Σ_κ  q[i,κ] · k[j,κ] · exp2( g[i,κ] − g[j,κ] )
窗口外 / 窗口内 j > i 的位置 = 0。
```

- `exp2(g[i]−g[j])`：逐 K 通道的衰减因子（`i==j` 不衰减，`j<i` 压弱早期 key）；
- Aqk **含对角线**；写回列 = `j % BT`（chunk 内位置），非对角块恒 0。

### 3b. Akk 的计算（key·β × gated-key 窗口打分）：`Akk[t, j'] = ⟨k[t]·β[t], k[j]⊙exp2(g[t]−g[j])⟩`

**数学公式：**

```
对 chunk 内行 token i 与列 token j（同窗口，且 j < i）：
  Akk[i, j] = Σ_κ  (k[i,κ]·β[i]) · k[j,κ] · exp2( g[i,κ] − g[j,κ] )
对角线（j==i）及窗口外 = 0。紧凑写回列 = j − s（0..BC-1）。
```

与 3a 仅两点差异：行向量用 `k·β` 而非 `q`；去掉对角线（严格因果 `j<i`）。
Akk 是后续 K3 三角求解的对角输入，故对角块必须为 0（上三角块由 inter_solve 填补）。

### 3c. 整块向量化：行/列因子拆分 + 两次大 dot（收敛 kernel 的写法）

**数学变换：**

```
exp2(g[i]−g[j]) = exp2(g[i]) · exp2(−g[j])          # 因子可分别预乘到行/列
⇒ 一个窗口 16×16 块（或整 chunk 64×64，块外补 0 后数学不变）：
   Aqk_full = ( qc⊙exp2(gc)·scale ) @ ( kc⊙exp2(−gc) )ᵀ     # [BT,K]@[K,BT]
   Akk_full = ( kc⊙βc⊙exp2(gc) )    @ ( kc⊙exp2(−gc) )ᵀ
```

**分块矩阵乘法图示**（一张 chunk 的 64×64 表；仅对角 4 个 16×16 块非零、块内下三角）：

```
       列 j → chunk 内 key 位置（0..63）
 行 i   ┌────────┬────────┬────────┬────────┐
 窗口0  │ ▣下三角 │    0   │    0   │    0   │
        ├────────┼────────┼────────┼────────┤
 窗口1  │    0   │ ▣下三角 │    0   │    0   │
        ├────────┼────────┼────────┼────────┤
 窗口2  │    0   │    0   │ ▣下三角 │    0   │
        ├────────┼────────┼────────┼────────┤
 窗口3  │    0   │    0   │    0   │ ▣下三角 │
        └────────┴────────┴────────┴────────┘
```

**逐 tile 分块代码：**

```
# grid = (cdiv(T,BT), B·(H//HM))；每 CTA 循环 HM 个 head；BT=64、BC=16
qc, kc, gc = load 整 chunk [BT, K]； betac = load [BT]          # 行/列两个打分共用
eg   = exp2(gc);  ene = exp2(−gc)                                # 两因子各算一次
Aqk_full = tl.dot( qc * eg * scale,  tl.trans(ke) )   # ke = kc*ene
kbe      = (kc * betac[:, None]) * eg
Akk_full = tl.dot( kbe,              tl.trans(ke) )

keep   = 对角块 ∧ (块内 r%BC ≥ c%BC)      # Aqk 下三角含对角
strict = 对角块 ∧ (块内 r%BC >  c%BC)      # Akk 严格下三角
Aqk_full = tl.where(keep,   Aqk_full, 0.0)
Akk_full = tl.where(strict, Akk_full, 0.0)

store Aqk + 行 offset + [0..BT)             # 无掩码连续写 [B,T,H,BT]
Akk_diag = tl.gather(Akk_full, 对角线列偏移, axis=1)   # [BT,BC] 对角块收拢
store Akk + 行 offset + [0..BC)             # 紧凑写 [B,T,H,BC]
```

**为何值置零而非 store-mask：** Ascend MTE 对非单调列地址的写会越界，Akk 紧凑列映射
（`col = c−(r//16)*16`）不能直接无掩码写；故 Akk 先在整 chunk 宽度内算出、把对角块
用 `tl.gather` 收拢成 `[BT,BC]` 再连续写，避免逐 lane 标量 store（aiv_scalar 头号瓶颈）。
同理 Aqk 直接全宽写、非对角块写 0 —— 与 torch ref 的零初值一致，免去 buffer 预清零。

## 4. 关键代码对应（`src/token_parallel_kernel.py`）

| 设计逻辑 | 本目录实现 |
|---------|-----------|
| Grid 定义: (B*T, H) | `token_parallel_triton` 中 `grid = (B * T, H)` |
| 全局 token 索引解析 | `_token_parallel_kernel`: `bos = (i_tg//T)*T; i_t = i_tg % T` |
| Sub-chunk 定位 | `i_c = i_t // BT; i_s = (i_t % BT) // BC; i_ts = i_c*BT + i_s*BC` |
| 当前 token q/k/g/beta 加载 | mask over `o_k` 载入 `qb/kb/gb`；`beta_v` 标量载入 |
| Key 预乘 beta | `kb = kb * beta_v` |
| Inner loop | `for j in range(i_ts, min(i_t+1, min(T, i_ts+BC)))` |
| Gate 衰减因子 | `kgj = kj * tl.math.exp2(gb - gj)` |
| 无效 K 维度掩码 | `kgj = tl.where(m_k, kgj, 0.0)` |
| Aqk dot product | `aqk = tl.sum(qb * kgj, axis=0) * scale` |
| Akk dot product (严格上三角) | `akk = tl.sum(kb * kgj, axis=0) * tl.where(j < i_t, 1.0, 0.0)` |
| Aqk 存储 | `tl.store(Aqk + bos*H*BT + i_t*H*BT + i_h*BT + (j % BT), aqk)` |
| Akk 存储 | `tl.store(Akk + bos*H*BC + i_t*H*BC + i_h*BC + (j - i_ts), akk)` |

## 5. torch 元算子实现（`token_parallel_torch`，性能/精度基准）

- 按 sub-chunk 批量化：对每 chunk 的每个 sub-chunk 位置 `a`，提取
  `qr/kr/gr/kbr = [B, NT, BC, H, K]` 切片；
- `dec[i,j] = exp2(gr[i] - gr[j])` → `kw[i,j] = kr[j] * dec[i,j]`；
- `aqk[i,j] = sum(qr[i] * kw[i,j])`，`akk[i,j] = sum(kbr[i] * kw[i,j])`；
- 乘因果掩码：`aqk *= tril(j<=i) * scale`；`akk *= tril(j<=i) * (~eye)`；
- 5D 连续写回 `Aqk5[:, :, a:a+BC, :, a:a+BC] = aqk.permute(0,1,2,4,3)` 后
  reshape 成 `[B, NT*BT, H, BT]`，截断到 `:T`。

它和 CPU 参考在 fp32 上逐元素一致（实测 max-diff ~1e-6），作为性能基准时
是整个 kda 算子图（两轮 broadcast + exp2 + 归约的中间张量往返）。

## 6. 精度 & 性能对比测试策略（配套 `run.py` / `testcases.csv`）

- 每个 case 固定 seed，输入分布：`q/k = randn`，`gk = randn*0.5`（log2 累积
  gate 量级较小），`beta = randn*0.1 + 1.0`（恒正、在对角线附近抖动）；
- 对每个 case 依次跑三个实现：
  1. **torch_npu 元算子** `token_parallel_torch` —— 精度基本准 + 性能基本准；
  2. **triton kernel** `token_parallel_triton` —— 被测对象；
  3. **CPU 参考** `token_parallel_ref`（逐 token 循环，ground truth）。
- 精度指标（两个都满足才 PASS）：
  - `max|triton - torch_npu| < 1e-2`（Aqk / Akk 分开算）；
  - `max|torch_npu - ref| < 1e-2` 且 `max|triton - ref| < 1e-2`。
  实测 max-diff 均为 fp32 累积噪声级（≤ ~1e-5）。
- 性能指标：预热 `--warmup`(默认 5) 次、各跑 `--repeats`(默认 30) 次，用
  `torch.npu.synchronize()` 包裹计时，输出每个 case 的
  **加速比 = torch_npu_time / triton_time**。
- CSV 中共 15 个 case，覆盖：完整 chunk / 尾 chunk 不满（`T=63/65/96/100/127/193/2562`）、
  单/多 head（`H=1/2/3`）、单/多 batch（`B=1/2`）、`K=32/64/128`。

## 7. 性能测试思路

- torch_npu 元算子：对每个 sub-chunk 各做一次 6 维 broadcast + `exp2` +
  `sum(-1)`，多次 kernel 启动与中间张量读写，是性能基准的下界参考；
- triton kernel：每 token 一个 CTA，单 kernel 完成遍历与双输出，无中间张量；
  循环从 L1 缓存读取同 sub-chunk 最近的 k/g，寄存器内完成 dot 积；
- 每 case 预热 5 次、计时 30 次（`torch.npu.synchronize()` 包裹），报告
  `torch_ms / triton_ms / speedup`。

## 附录 A: 关键常量速查

| 常量 | 值 | 用途 |
|------|-----|------|
| `BT` | 64 | Chunk 大小 |
| `BC` | 16 | Sub-chunk 大小（NC = BT/BC = 4） |
| `BK` | `next_power_of_2(K)` | K 维 block/pad |
| `BH` | 1 | 每 CTA head 数 |
| `num_warps` | 1 | 每 CTA warp 数（与真实 kernel 一致） |

## 附录 B: 与真实 kernel 的差异对照

| 项 | `python/.../fla/chunk_intra_token_parallel.py` | 本目录独立实现 |
|----|------------------------------------------------|----------------|
| VARLEN / `cu_seqlens` 二分查找 | 支持 | 只做固定长度 `bos=(i_tg//T)*T` |
| autotune (`BH in [1]`) | 有 | 无（固定 BH=1 直接编译） |
| 无 NPU 环境 | 无法运行 | `token_parallel_ref` 退化为纯 CPU 参考 |
| 依赖 | sglang 包 | 仅 torch + torch_npu + triton |