# KDA 6 算子全解析：从数学定义到 Triton Kernel 实现与测试

> 本文档是 KDA（Kimi Linear Delta Attention）chunked attention 6 个算子的**完整学习
> 指南**：从算法数学定义出发，结合 Ascend（A5/950PR）硬件特性，讲解 6 个 triton
> kernel 的核函数设计思路、逐轮优化技术（含性能数据与踩坑记录），最后落到统一
> 测试框架与测试结果。适合想从全貌理解这套实现的人。
>
> 配套材料索引见文末 §9。

---

## 1. 总览：6 个算子是什么

KDA 是 Kimi 的线性 Delta Attention 变体。其 chunkwise forward 把一次 attention
分解为 6 个可独立实现/验证的算子（K1..K6），数据流如下（`unified/README.md`）：

```
g       = K1(x, A_log, dt_bias)                          # [B,T,H,K]  gate cumsum
Aqk_d, Akk  = K2(q, k, g, beta, scale)                   # 块内对角线得分
Aqk_nd, Akk_inv = K3(q, k, g, beta, Akkd=Akk, scale)     # 块间得分 + 逆
w, u, kg = K4(k, v, beta, A=Akk_inv, gk=g)               # 解耦表示
h, v_new = K5(kg, w, u, gk=g, initial_state, idx)        # 跨 chunk 状态递推
o       = K6(q, v_new, g, Aqk=Aqk_d+Aqk_nd, h=h, scale)  # 输出
```

| # | 算子 | 别名 | 一句话职责 | 主要计算形态 |
|---|---|---|---|---|
| K1 | gate_chunk_cumsum | Gate | gate 激活 + chunk 内前缀和 | cumsum（向量） |
| K2 | token_parallel | 块内注意力 | 同 sub-chunk 的 Aqk/Akk 对角线块 | 小 dot ×4 |
| K3 | inter_solve | 块间逆 | 跨 sub-chunk 得分 + 块下三角逆 | 串行 dot 链 |
| K4 | recompute_w_u | 解耦 | A@(k·β·g) / A@(v·β) | 大 dot |
| K5 | delta_rule_h | FwdH | Delta Rule 状态跨 chunk 递推 | 序列 dot 链 |
| K6 | gla_output | Finalize | 跨块 + 块内输出合并 | 大 dot + 小 dot |

**目标 case**（全部性能数据以此为准）：`B=1, T=16384, H=96, K=V=128`（case_id
`D_KV128_H96_T16384`），BT=64（chunk 大小）、BC=16（sub-chunk）。

---

## 2. 数学定义（逐算子）

### K1 gate_chunk_cumsum

```
gate[t,s] = -exp(A_log[h]) · softplus(x[t,s] + dt_bias[h,s])     # softplus 阈值 20
out       = scale · chunk_local_cumsum(gate)                      # chunk 内从 0 重新开始
```
- 输入 `x [B,T,H,K]`（raw gate）、`A_log [H]`、`dt_bias [H*K]`（可选）
- 输出 `[B,T,H,K]`，`scale = RCP_LN2 = 1.442695…`（把 ln 空间换算到 **log2 空间**，
  供 K2/K3 的 `exp2(g[i]-g[j])` 直接使用）
- 关键语义：**chunk 局部前缀和**，跨 chunk 不传递（对应 chunked attention 的
  每 chunk 独立 gate 归一）

### K2 token_parallel（块内得分）

```
Aqk[i,j] = <q[i], k[j]·exp2(g[i]-g[j])> · scale        (j ≤ i，同 sub-chunk)
Akk[i,j] = <k[i]·β[i], k[j]·exp2(g[i]-g[j])>          (j < i，同 sub-chunk，去对角线)
```
- 输入 `q/k/g [B,T,H,K]`、`beta [B,T,H]`、`scale=K^-0.5`
- 输出 `Aqk [B,T,H,BT]`（列=j%BT，含对角线）、`Akk [B,T,H,BC]`（列=j-i_ts，严格下三角）
- 数学变换（Route A 引入）：`exp2(g[i]-g[j]) → exp2(g[i])·exp2(-g[j])`，把逐对
  指数计算拆成两个逐元素乘，使整个 sub-chunk 变成一次 batched dot：
  `Aqk = (q·exp2(g)) @ (k·exp2(-g))^T`（max_diff 7.6e-6，可接受）

### K3 inter_solve（块间得分 + 逆）

三阶段：
- **Phase 1**（6 对 i>j 非对角块）：
  `Akk_ij = (K_i·exp2(G_i−G_i[last])) @ (K_j·exp2(G_i[last]−G_j))^T · β_j`；
  `Aqk_ij` 同式换 Q、乘 scale（`G_i[last]` = 子块 i 末尾 token 的 gate）
- **Phase 2**：对角块下三角逆 `D_inv = (I − strict_tril(D))⁻¹`（逐行前向替换）
- **Phase 3**：链式合并 `Ai_10 = −Ai_11 @ Akk_10 @ Ai_00`（3 层共 6 个非对角）
- 输入 `q/k/g`、`beta`、`Akkd`（K2 输出的对角块）、`scale`；输出 `Aqk_nd [B,T,H,BT]`、
  `Akk_inv [B,T,H,BT]`（下三角逆，上三角 0）

### K4 recompute_w_u（解耦表示）

```
w  = A @ (k·β·exp2(gk))        # Key 解耦表示（可跨 chunk 递推）
u  = A @ (v·β)                 # Value（不受 gate 影响）
kg = k·exp2(gk_last − gk)      # 时间对齐，gk_last = chunk 内最后有效 token
```
- 输入 `k [B,T,H,K]`、`v [B,T,H,V]`、`beta`、`A=Akk_inv [B,T,H,BT]`、`gk`；输出三个 `[B,T,H,K]`
- 这是把 K5 的逐 chunk 递推从"每 chunk 重新读原始 k/v"变成"每 chunk 只读解耦后的
  w/u/kg"，是 chunked attention 的关键代数重组

### K5 delta_rule_h（状态递推）

```
for c in 0..NT-1:                      # chunk 间串行（状态跨 chunk 传递）
    ① h[c] = state                     # 快照（供 K6 读）
    ② v_new = u − w @ state^T          # Delta Rule 残差
    ③ state *= exp2(gk_last)           # per-channel 衰减（log2 空间）
    ④ state += k^T @ v_new             # 外积累加
写回 final state（in-place 更新 initial_state）
```
- 输入 `kg/w [B,T,H,K]`、`u [B,T,H,V]`、`gk`、`initial_state [N,H,V,K]`、`indices [B]`
- 输出 `h [B,NT,H,V,K]`、`v_new [B,T,H,V]`；约束 K==V、K≤256
- **这是全链唯一 chunk 间有依赖的算子**，串行性是性能关键

### K6 gla_output（输出合并）

```
o[t] = o_cross[t] + o_intra[t]
o_cross: q_gated = q·scale·exp2(g)； o_cross = q_gated @ h^T     # 跨块，K 维循环累加
o_intra: o_intra = where(下三角, Aqk, 0) @ v_new                 # 块内
```
- 输入 `q`、`v_new`（K5 输出）、`g`（K1 输出）、`Aqk`（=Aqk_d + Aqk_nd）、`h`（K5 输出）
- 输出 `o [B,T,H,V]`

---

## 3. 目标硬件特性：Ascend A5 与 triton-ascend

所有 kernel 的设计都受以下硬件特性约束（这些是在 6 个 kernel 的迭代中**实测**
总结出来的，不是纸面规格）：

### 3.1 硬件结构

- **AI Core 三单元**：Cube（矩阵乘，fp16/fp32）、Vector（逐元素运算，AIV）、
  Scalar（地址/控制/标量运算，AIC）。三者独立流水，**任一饱和都是瓶颈**。
- **片上 UB**（统一缓冲）：数据在 HBM↔UB 间经 MTE 搬运，tile 必须能放进 UB。
  A5 实测可用 UB 约 248KB（2031616 bits），fp32 tile 64×128=32KB。
- **grid 展平上限 65535**：triton-ascend 把 3D/2D grid 展平为 1D 后，总 CTA 数
  不能超过 65535，否则启动报 `ERR00100 value 65536 for parameter coreDim is invalid`。

### 3.2 实测踩坑清单（按 kernel 归类）

| # | 坑 | 现象 | 规避/修复 |
|---|---|---|---|
| 1 | grid 展平 > 65535 | kernel 无法启动 | 增大 tile（K1 BS 32→128）、head-merge 降 CTA 数 |
| 2 | 标量管线饱和 | `aiv_scalar/aic_scalar_ratio` 高企（阈值 0.10） | 去 Python 循环（tl.dot 化）、去 boundary_check、constexpr 分派、掩码提循环外 |
| 3 | MTE 列寻址越界 | Akk 非单调列映射 → aicore exception | 满宽连续写 + driver `torch.gather` 收拢（K2） |
| 4 | 高 CTA 数触发 CANN UB 分配 bug | aicore exception | head-merge 24576→1536 CTA（K2/K3/K4/K6） |
| 5 | `tl.gather` 对 dot 输出不可靠 | 数值错 0.32 且慢 3×（triton-ascend 3.2.1） | 回退 driver `torch.gather`（K2，hm3→hm2） |
| 6 | hivm-plan-memory 对 `trans(dot 输出)` 失败 | 报 ub overflow（假象） | 外积改写为输入侧转置 `dot(trans(b_v), b_k)`（K5） |
| 7 | flat reshape store 的 expand_shape bug | CANN 9.1 报 dim 2048≠4096 | 统一 2D 手动指针 store（K5 K=64 路径） |
| 8 | block_ptr 的 boundary_check 标量退避 | 标量比率↑ | 仅保留必要维的 check；T%BT==0 时 constexpr 免检（K4） |
| 9 | `tl.range(num_stages)` 静默忽略 | 流水线 no-op | R5 接线后 NS=3 才生效（K5 9.51→9.11ms） |
| 10 | fp16 dot 被 upcast 回 fp32 | cube 未启用（K6 R4 负结果） | 保持 fp32 路径 |
| 11 | torch_npu 的 `torch.arange` 垃圾值 | B≥2 时 int32 错值 | CPU 创建后 `.to(device)`（bench 框架） |
| 12 | 800MB memset / 402MB scratch 每调用 | 每 kernel 启动代价毫秒级 | `torch.empty` 缓冲 + 无掩码 store + kernel 内收拢 |

**核心认知**：在这套实现里，真正的瓶颈几乎从来不是 cube 吞吐（各 kernel
cube 利用率大多 <20%），而是**标量管线（寻址/掩码/循环控制）与访存**。因此
优化主线是"削减每 CTA 的固定标量开销"（head-merge、免掩码、constexpr 分派）与
"放大每次 dot 的粒度"（整 K 单 tile、BV=V）。

---

## 4. 六个 kernel 的核函数设计（结合代码）

### 4.1 K1 gate_chunk_cumsum（`src/gate_kernel.py`）

**设计**：`grid = (cdiv(K,BS), num_chunks, B*H)`，每个 CTA 处理一个 `[BT=64, BS=128]`
tile（行=时间 chunk，列=通道），对 tile 做 `tl.cumsum(b_gate, axis=0)`（chunk 局部
前缀和）。BS 的演进就是 grid 超限与标量摊薄的故事：

| 轮 | 改动 | 展平 grid | 耗时 | 动机 |
|---|---|---|---|---|
| 基线 | BS=32 | 98304（超限） | 不支持 | — |
| R1 | BS 32→64 | 49152 | 4.32ms | grid 降到 65535 内 |
| R2 | BS 64→128 | 24576 | ~1.85ms | block 数减半、每 block 密度翻倍 |

关键代码（尾 chunk 补零安全性的精妙之处）：

```python
b_gate = -tl.exp(b_a) * _softplus_fwd(b_s)     # gate = -exp(A_log)·softplus(x+bias)
b_gate = tl.where(masks, b_gate, 0.0)          # cumsum 前清无效行
b_o = tl.cumsum(b_gate, axis=0)                # 沿 axis=0 向后累加
```
尾部补零不会污染头部有效行的前缀和（cumsum 单向累加），因此可以无掩码地大块处理。

**硬件考量**：UB 上限实测——BS=128/BT=128 的 128×128 fp32 tile 报
`ub overflow: 2625536 bits > 1572864 bits`，给出 UB 预算的直接证据。

### 4.2 K2 token_parallel（`src/token_parallel_kernel.py`）

**设计演进是"标量→向量→粒度"三连跳**：

- **基线**：`grid=(B*T, H)`，每 (token, head) 一个 CTA + Python `for j` 循环 →
  grid 1.57M 超限 + scalar_ratio 0.38（FAIL）
- **Route B**：chunked grid `(B·cdiv(T,BT), H)` + block_ptr + scalar fix →
  per-token 3.74us（5.38×），但 scalar 0.30 仍 FAIL（for 循环是根源）
- **Route A**：`exp2(g[i]-g[j]) → exp2(g[i])·exp2(-g[j])` 数学变换，sub-chunk 一次
  `[16,128]@[128,16]` 小 dot 批量算 → scalar 0.059（达标）但 **0.91× 变慢**
  （cube_util 仅 7.87%——矩阵太小不进 cube）
- **Route C + head-merge（最终）**：`grid=(cdiv(T,BT), B*H//16)`，每 CTA 串行 16 个
  head，CTA 24576→1536；标量裁剪（去 K 维 mask、scale 折叠进 pre-dot q、exp2 各算
  一次、掩码循环外预计算）+ `torch.empty` 缓冲。**10.5→8.66ms**
- **R3.5 tl.gather 收拢 Akk**：Akk 对角块由 driver `torch.gather`（~2ms/调用，读
  402MB 满宽 scratch）改为 kernel 内 `tl.gather(axis=1)` → 集成 10.7→~9.4ms。
  **⚠️ 但这在 CANN 9.1/triton-ascend 3.2.1 下数值错误（max_diff 0.32）且慢 3×**，
  本分支已回退 hm2 路径（满宽写 + driver gather），hm3 标注 DEPRECATED。

最终结构（hm2，head 循环内）：
```python
for hh in tl.range(HM, num_stages=NS):            # HM=16 head 合并
    qe = qc * eg * scale                          # scale 折叠进 pre-dot
    ke = kc * eneg
    Aqk_full = tl.dot(qe, tl.trans(ke))           # 整 chunk 大 dot [64,128]@[128,64]
    kbe = (kc * betac[:, None]) * eg
    Akk_full = tl.dot(kbe, tl.trans(ke))
    Aqk_full = tl.where(keep, Aqk_full, 0.0)
    tl.store(base_aqk + row_akk + col_bt, Aqk_full)          # Aqk 直接写 [B,T,H,BT]
    Akk_full = tl.where(strict, Akk_full, 0.0)
    tl.store(base_akk + row_akk + col_bt, Akk_full)          # Akk 满宽写 scratch
# driver: Akk = _gather_akk_diag(scratch)  # 对角块收拢到 [B,T,H,16]
```

**硬件考量**：Akk 的紧凑列映射（`c-(r//16)*16`）会让 MTE 列地址非单调 → 越界，
所以用"满宽写 + 收拢"两步；CTA 数量级（24576 vs 1536）直接决定标量脚手架总量。

### 4.3 K3 inter_solve（`src/inter_solve_kernel.py`）

**设计**：head-merged `grid=(cdiv(T,BT), B*(H//16))`，每 CTA 三阶段融合（Phase 1/2/3
在一个 kernel 内完成）。关键优化是**截断逆的幂级数展开**：

```
(I−L)⁻¹ = (I−L)(I+L²)(I+L⁴)(I+L⁸)     # npow=3，6 个 [64,64] dot
```
替代逐行前向替换（或 23 个链式小 dot）。npow 扫描：5→19.2ms、4→16.2ms、**3→14.4ms
最优**、2→18.0ms 且精度 7.1e-5。代码形态：

```python
b_pow = tl.dot(b_pow, b_pow)             # L^2, L^4, L^8
b_inv = tl.dot(b_inv, b_I + b_pow)       # 逐级累乘
```

优化轮次：R1 去 15 处 `input_precision="ieee"`（Ascend 上限制优化）；R2 head-merge +
截断逆 → **66.5→14.5ms**（~4.6×，其中高 CTA 数 aicore exception 也是靠 HM 规避）；
R3 empty 缓冲 + 无掩码 store；R4 证伪 fp16（cube 加速未兑现）；R5 收敛 NP=3+HW=16+nw=4
= **14.37ms**。

**硬件考量**：aic_scalar≈49%、cube≈12.3%——瓶颈是**标量寻址 + 逆 dot 串行依赖
延迟**（非吞吐）；triton 3.2.0 无 `tl.extract_slice`/动态索引，Phase 2 的寄存器
预加载方案不可行。

### 4.4 K4 recompute_w_u（`src/recompute_w_u_kernel.py`）

**设计**：`grid=(NT, cdiv(B*H,16))`（HM=16），每 CTA 处理一个 (chunk, head)：
公共加载 `A_inv [BT,BT]`、`beta [BT]` 留寄存器，随后 V 维（u）+ K 维（w）两个
dot（kg 顺带算，避免重复访存）。

优化主线是**标量退避**：
- 基线：9+ 处 block_ptr 全带 boundary_check + i32 掩码 → aiv_scalar 0.489、cube 15.5%
- **Route A**：`BK=K, BV=V` 单 tile 直通（无 tile 循环无掩码）+ `T_FULL` constexpr
  分派（T%BT==0 全免 boundary_check）+ nw=4 → aic_scalar **0.359→0.094**（−74%），
  1.68×
- **R5 HM=16**：CTA 24576→1536 → **6796→4514us（1.51×）**，max_diff=0.0（bitwise）

```python
b_A  = tl.load(p_A + (base + o_bt[:, None]) * s_A + o_bt[None, :])  # [BT,BT] 公共
b_u  = tl.dot(b_A, b_vb, input_precision=DOT_PRECISION)             # v·β
b_kg = b_k * tl.math.exp2(b_gn[None, :] - b_gk)                     # 时间对齐
```

**硬件考量**：512B 行对齐提升向量利用率；nw>4 时 warp 调度开销>收益；tf32 在
triton-ascend 上等价 ieee。

### 4.5 K5 delta_rule_h（`src/delta_rule_h_kernel.py`）

**设计**：全链唯一 chunk 间串行的算子。`grid=(cdiv(V,BV), B*H)`，**BV=V**（整 V 驻留
一个 CTA，BV 扫描 32→40ms / 64→20ms / 128→10ms 单调最优）；状态寄存器
`b_h [BV,K]` fp32 常驻；主循环每迭代：快照 store → Delta Rule dot → v_new store →
gk 衰减 → 外积更新：

```python
for i_t in tl.range(NT, num_stages=NS):              # NS=3 软件流水线
    _store_h_full(h, i_t * stride_h, i_v * BV, K, BV, b_h)
    b_v = tl.dot(b_w, tl.trans(b_h).to(b_w.dtype))   # Delta Rule: w @ h^T
    b_v = tl.load(p_v, boundary_check=(0,)) - b_v    # u − w@h^T
    ...
    b_h *= _exp2(b_gk_last)[None, :]                 # per-channel 衰减
    b_h += tl.dot(tl.trans(b_v), b_k2)               # 外积（输入侧转置，见下）
```

优化轮次：R1 三路线对照（2D store 是反模式：aiv_scalar 0.436→0.472）；R2 BV=V
4×（40→10ms）；R3 整 K 单 tile（4 dot→2 dot/chunk，9.78→9.09ms，隔离测出 memory
下界 3.02ms）；R5 `tl.range(num_stages)` 接线（此前 num_stages 被静默丢弃！
9.51→9.11ms）。

**本轮修复的两个 CANN 9.1 编译问题（详见 TEST_REPORT）**：
1. `b_h += trans(dot(k, b_v))` 的 dot 输出转置+累加链 → hivm root-alloc 失败
   （ub overflow 误报）→ 改写为 `dot(trans(b_v), b_k2)`（行主 `[BT,K]` 加载 +
   输入侧转置，数学等价）。此形态 11.5ms（+25% vs 基线 9.15ms，三种替代实验
   21.8/22.3ms 更差，为当前最优可用方案）。
2. K=64 的 flat reshape store → expand_shape bug → 统一 2D 手动指针 store。

```python
# _store_h_full: 通用 2D store（不用 flat reshape —— triton-ascend MLIR bug）
offs_v = v_start + tl.arange(0, BV)
offs_k = tl.arange(0, K)
ptr = base + chunk_offset + offs_v[:, None] * K + offs_k[None, :]
tl.store(ptr, b_h.to(base.dtype.element_ty))
```

### 4.6 K6 gla_output（`src/gla_output_kernel.py`）

**设计**：`grid=(cdiv(V,BV=128), NT, B*H//HM)`，`for hh in tl.range(HM=16)` head
合并；**全部 loop-invariant 的掩码/偏移提循环外**；`b_o = tl.zeros([BT,BV])` 必须
在 hh 循环体内重声明（每 head 独立累加器）。跨块部分按 K 维循环累加
`tl.dot(b_qg, trans(b_h))`，块内 `tl.dot(where(下三角, b_A, 0), b_v)`。

优化轮次：R1 手动指针+fp32 mask 胜出（block_ptr 有 IR 膨胀成本，1.57×）；R2
跨块 BK 32→128 合并 K 循环（8.2→6.8ms）；R4 ai-core 隔离 profile **证伪内存下界**：
aiv_scalar **97.8%**（地址生成饱和）而非 mte2 受限；R5 head-merge HM=16 →
**6.94→4.75ms**，代价转移：aiv_scalar 97.8→60.9%，新瓶颈 aic_mte2 92.5%。

**硬件考量**：这个 kernel 是"标量饱和"最典型的案例——每 CTA 固定 setup（arange
偏移、因果 mask、指针 base）与 head 无关却每 head 重算，HM=16 摊薄 16×；hm32 无
进一步收益（4.74ms）；block_ptr 重写 7.0ms **证伪**"手动 int64 寻址耗标量"——
耗标量的是每 CTA setup 而非寻址形态。

---

## 5. 跨 kernel 的优化技术清单

| 技术 | 应用 | 收益量级 | 原理 |
|---|---|---|---|
| **head-merge（HM=16）** | K2/K3/K4/K6 | 1.5×~4.6× | 每 CTA 固定标量 setup（掩码/寻址/指针）与 head 无关，合并 HM 个 head 摊薄；同时降 CTA 数规避 aicore exception |
| **标量削减** | 全部 | 累计 ~2× | 去 Python 循环（tl.dot 化）、去 boundary_check（constexpr 分派）、掩码/偏移循环外预计算、scale 折叠进 pre-dot |
| **整 K 单 tile / BV=V** | K4/K5/K6 | 1.3×~4× | 消除 tile 循环与重复加载，放大 dot 粒度（K5: 4 dot→2 dot；K6: 4 小 dot→1 大 dot） |
| **数学变换** | K2/K3 | — | `exp2(g[i]-g[j]) → exp2(g[i])·exp2(-g[j])` 把逐对计算变成 batched dot；截断逆幂级数替代串行前向替换 |
| **无掩码 store + empty 缓冲** | K2/K3/K4 | 毫秒级/调用 | 输出缓冲按 NT*BT 补齐后 store 免 mask，掩码处写 0 与零初值等价；省 800MB memset kernel |
| **tl.range 软件流水线** | K5 | 9.51→9.11ms | 预取下一 chunk 的 w/u/k 掩盖序列链 dot 延迟（注意 num_stages 必须显式接线） |
| **满宽写 + driver 收拢** | K2 | — | MTE 列寻址越界时换布局写再收拢；tl.gather 不可靠时回退 torch.gather |

---

## 6. 统一测试框架（`design/unified/`）

### 6.1 设计

- **cases_meta.json**：106 个 case 的元信息（A 组 T 主扫描 72 + B 组非 2 幂边界 20 +
  C 组多 batch 8 + D 组 K=V 扩展 6），`case_id → {B,T,H,K,V,group,desc}`；张量由
  `bench.py::_gen_case_inputs` 固定种子（20260815）即时生成，避免多 GB CSV。
- **数据流**：每个 case 先用 torch 链算共享中间量（K1..K6 的 torch 元算子版），
  再对 K1..K6 各自用完全相同输入跑 `K_torch` vs `K_triton` 得 max_diff。
- **模式 A（正确性）**：逐 case 6 kernel 精度对比 → `correctness.csv`
  （判定 `max_diff < 1e-2`）。
- **模式 B（msprof 性能）**：性能唯一口径 = msprof `op_summary.csv` 的
  `Task Duration(us)`（设备侧 kernel 时间，**不做 wall-clock**）。
- **marker 分段**：微 kernel `_kda_bench_marker`（1 元素 store）做 (case,kernel)
  分界，`per_case_profile.py` 按 op_summary 行序切段：段内 triton op 名开头的行 =
  triton 时间，其余 = torch_npu 拼接时间；`--mean` 输出每次调用均值。

```bash
python3 bench.py --start 105 --limit 1                       # 模式 A：目标 case
bash run_cpu.sh --msprof ./prof_target_d3 --start 105 --limit 1 \
     --repeats 5 --warmup 2                                  # 模式 B：msprof 采集
python3 per_case_profile.py --latest-dir ./prof_target_d3 --mean   # → results.csv
python3 analyze_results.py --pivot-case D_KV128_H96_T16384        # 加速比矩阵
```

### 6.2 已知环境坑（框架内已处理）

- NPU 上 `torch.arange` 多种 B 产生垃圾值 → CPU 创建后 `.to(device)`
- K5 in-place 污染 initial_state → 独立 init + backup 复位
- 持续负载下随机 `aclnnMul` 崩溃（驱动瞬时故障）→ 分批跑 + 重启容器
- 全量 106 case 分批采集用 `run_all_msprof_local.sh`（T≥65536 单 case 一批）

---

## 7. 测试结果（目标 case，2026-08-31）

### 7.1 triton 6-kernel（fp32，msprof 每次调用均值）

| Kernel | torch_us | triton_us | speedup | max_diff | vs 官方基线 |
|---|---|---|---|---|---|
| K1 | 9844.8 | **1379.96** | 7.13x | 1.14e-05 OK | −27% |
| K2 | 132299.5 | **5620.98** | 23.54x | 1.49e-07 OK | −40% |
| K3 | 41233.1 | **10692.67** | 3.86x | 1.49e-07 OK | −26% |
| K4 | 19477.9 | **3609.17** | 5.40x | 1.30e-04 OK | −20% |
| K5 | 696780.8 | **11425.37** | 60.99x | 1.49e-08 OK | **+25%** |
| K6 | 14191.0 | **3914.44** | 3.63x | 9.33e-03 OK | −47% |

总 triton ≈ **36.64ms**，正确性 6/6 PASS。K1/K2/K3/K4/K6 均快于官方基线，
K5 因 CANN 9.1 编译问题（§4.5）慢 25%。

### 7.2 Ascend C 融合算子对照（vllm-ascend ChunkKdaFwd）

| 指标 | triton 6-kernel（fp32） | Ascend C 融合（fp16） | Ascend C 融合（bf16） |
|---|---|---|---|
| 总耗时 | 36.64ms | 43.67ms | 37.22ms |
| 正确性 | max_diff<1e-2（6/6） | max_abs 1.9e-6（T=1024） | 同左 |

同数量级、triton 分拆版略快（且是 fp32），两者均 memory/流水线受限；融合算子
bf16 比 fp16 快 15%（带宽敏感，Kimi K3 部署有利）。

### 7.3 遗留问题

1. K5 +25%（CANN 9.1 下三种替代形态 11.5/21.8/22.3ms，输入侧转置为最优）
2. K6 精度临界 9.33e-3（HM=16 代价）；A 组小 T case（T≤2048）K6 超阈值 FAIL
3. K4 精度 1.30e-4（基线 0.0）未定位
4. 全量 106 case 未跑（仅目标 case）

---

## 8. 从"看代码"到"会优化"的要点提炼

1. **先分清瓶颈单元**：Cube 吞吐 / Vector / Scalar / 访存（mte）——msprof 的
   `aic_mac / aic_mte2 / aiv_vec / aiv_scalar` 指标直接告诉你该往哪个方向优化。
   本套实现里标量管线是第一瓶颈，别急着调 dot 形状。
2. **CTA 粒度是标量成本的放大器**：per-CTA 固定开销（掩码、地址、指针）与
   tile 大小无关，head-merge / 放大 tile 是最直接的摊薄手段。
3. **数学变换 > 编译器技巧**：`exp2(g[i]-g[j])` 拆分、截断逆幂级数、外积
   输入侧转置——这些代数重组同时改善数值与编译路径。
4. **编译器行为要实测**：`tl.range(num_stages)` 静默忽略、`tl.gather` 对 dot
   输出算错、`trans(dot)` 触发 hivm 失败、flat reshape 的 expand_shape——都是
   在特定 CANN 版本下才暴露，最小复现实验（如 `tmp_k2_gather_test.py`）是定位
   利器。
5. **性能口径只信 msprof**：设备侧 kernel 时间（Task Duration），不做 wall-clock；
   集成 vs 隔离数据要分开标注（集成含调度/拼接）。

---

## 9. 参考文档索引

| 内容 | 路径 |
|---|---|
| 各算子顶层定义 | `design/Kernel1_GateChunkCumsum.md` ~ `Kernel6_GLAOutput.md` |
| 各算子实现与优化历史 | `design/<op>/DESIGN.md`、`README.md`、`OPTIMIZATION_LOG.md` |
| 各算子 kernel 源码 | `design/<op>/src/<op>_kernel.py` |
| 统一测试框架 | `design/unified/README.md`（用法）、`ANALYSIS.md`（结果分析） |
| 目标 case 测试报告（2026-08-31） | `design/unified/TEST_REPORT_D_KV128_H96_T16384.md` |
| Ascend C 融合算子测试 | `chunk_kda_fwd_fused_bench.md` |
| KDA 算法背景 | `KDA_KERNELS_EXPLAINED.md` |
