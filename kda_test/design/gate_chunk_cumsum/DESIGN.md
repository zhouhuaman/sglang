# Kernel 1 设计文档: 独立 Gate + Chunk Cumsum 算子

> 本文档是 `kda_test/design/Kernel1_GateChunkCumsum.md` 的**独立子集重构**。
> 与原始文档的差异:
> - 不再假设 VARLEN / safe-gate / chunk_indices 等扩展路径,只保留
>   `B,T,H,K` 固定长度 + `Standard Gate + HAS_SCALE` 的最小闭环;
> - 行号改为引用本目录的 `src/gate_kernel.py`;
> - 增加「精度测试策略」和「性能测试思路」两节,配套本目录测试驱动。

---

## 1. 输入输出定义

### 1.1 输入张量

| 参数名 | Shape | Dtype | 含义 |
|--------|-------|-------|------|
| `x` (raw_gate) | `[B, T, H, K]` | fp32/bf16 | 未经激活的原始门控值，`K` 为每个 head 的通道维（SE = S） |
| `A_log` | `[H]` | fp32 | 每个 head 的对数尺度参数，控制衰减强度 |
| `dt_bias` | `[H*K]`（可选） | fp32 | 每个 head 每通道的偏置，平铺为 `[H*K]` |
| `scale` | 标量（可选） | fp32 | 输出缩放因子，实际调用时固定为 `RCP_LN2=1.4426950216293335` |
| `chunk_size` | 标量 | int | Chunk 大小 `BT=64`，必须为 2 的幂 |

编译期常量：`H`（head 数）、`K`（通道数=SE）、`BT`（chunk 大小）、`BS`（通道 tile 大小=32）、`HAS_BIAS`、`HAS_SCALE`。

### 1.2 输出张量

| 参数名 | Shape | Dtype | 含义 |
|--------|-------|-------|------|
| `o` (g) | `[B, T, H, K]` | fp32 | 门控激活 + chunk 内累积求和结果，已转换为 log2 空间 |

## 2. 分核并行策略（Grid 拓扑）

```
Grid = (cdiv(K, BS), NT, B * H)
        ~~~~~~~~~~   ~~  ~~~~~
          |            |      |
       通道维分块   chunk数  所有 (batch, head) 对
```

- `cdiv(K, BS)`：K 按 `BS=32` 切分得到的 S 维 tile 数；
- `NT = cdiv(T, BT)`：时间维 chunk 总数；
- `B * H`：每个 (batch, head) 组合一个平面。

每个 CTA 处理一个 `(时间 chunk i_t, S-tile i_s, (b,h))`，加载 `[BT, BS]` 的 tile
（行=时间、列=通道），独立完成门控激活 + chunk 内前缀和。

### 程序 ID 映射

```
i_s  = program_id(0)   S 维 tile 索引 (0 .. cdiv(K,BS)-1)
i_t  = program_id(1)   chunk 索引 (0 .. NT-1)
i_bh = program_id(2)   (batch, head) 联合索引 (0 .. B*H-1)
i_b  = i_bh // H       batch 索引
i_h  = i_bh %  H       head 索引
```

## 3. 计算思路

> 目标：每个 token 的逐通道累积衰减 `gk`（log2 空间）。它是一条 **激活 → chunk 局部
> 前缀和 → 缩放** 的逐元素流水线；同一个 (b,h) 的相邻通道/相邻时间完全独立，因此
> kernel 把它铺成二维 tile（行 = chunk 内时间、列 = 通道块）一次向量化做完。

### 3a. gate 的计算（逐元素激活）：`gate[t,k] = -exp(A_log[h]) * softplus(x[t,k] + dt_bias[h,k])`

**数学公式：**

```
gate[t, k] = -exp(A_log[h]) * softplus(x[t, k] + dt_bias[h*K+k])
softplus(u) = log(1 + exp(u))，u ≥ 20 时取 u（防 exp 溢出）
```

逐元素含义：
- `dt_bias[h*K+k]` 每个 (head,通道) 一个偏置 → 使 gate 的"零点"逐通道可调；
- `softplus` 保证括号内为正；`A_log[h]` 每 head 一个尺度，控制整 head 衰减强度；
- **负号** ⇒ gate 恒为负 ⇒ 前缀和后 gk 单调递减，衰减随 token 数加深。

**Tile 布局示意**（本算子无矩阵乘；kernel 把"一条 (b,h,通道块) 的 64 步前缀和"
向量化成 [BT 行时间 × BS 列通道] 的二维 tile）：

```
 一个 CTA 处理的时间 chunk c（首 token tc=c*BT）           列 = 一个通道块 S-tile
        ┌───────────────────────────────────────┐        BS = 128（真 kernel 常量）
 行 t0   │  x[t0, s0]  x[t0, s0+1] ... x[t0,s0+BS-1]│ ──┐
 行 t0+1 │  x[t0+1, s0]           ...              │   │ +dt_bias[h, s0..s0+BS]（按列）
 行 t0+2 │  ...                                    │   │ softplus
  ...    │                                        │   │ ×(-exp(A_log[h]))（按行标量）
 行 t0+63│  x[t0+63, s0]          ...              │ ──┘
        └───────────────────────────────────────┘      → gate[BT, BS]
grid = (cdiv(K,BS), NT, B*H) ：program(0)=S-tile，(1)=chunk，(2)=(b,h)
```

**逐 tile 分块代码：**

```
i_s, i_t, i_bh = tl.program_id(0..2)          # (通道块, chunk, batch*head)
b_s   = load(x + 偏移(tc..tc+63, s0..s0+BS-1))   # [BT, BS]  raw gate
b_s  += load(dt_bias + h*K + s0..s0+BS-1)[None, :]  # 加偏置，按列广播
b_a   = load(A_log + h)                        # 每 head 一个标量
b_gate = -tl.exp(b_a) * _softplus(b_s)         # [BT, BS]  gate 激活
b_gate = tl.where(行末越界, 0.0, b_gate)        # 尾 chunk 无效行清零（见 3b）
```

### 3b. gk 的计算（chunk 局部前缀和）：`gk[t,k] = cumsum_{chunk 内}(gate)[t,k]`

**数学公式：**

```
对 chunk c（首 token tc = c*BT）与通道 k：
  gk[tc+i, k] = gate[tc, k] + gate[tc+1, k] + ... + gate[tc+i, k]     (i = 0..BT-1)

跨 chunk 不传递：每个 chunk 从 0 重新开始（chunk N 首行 = gate[N*BT]，不带上个 chunk 的累计）。
```

> 前缀和逐通道独立、逐 chunk 独立，因此没有跨 chunk / 跨通道的任何串行依赖 —— 每个 CTA
> 只需要自己那一块 [BT, BS]，这也是能一步 `tl.cumsum(axis=0)` 的原因。

**逐 tile 分块代码（复用 3a 的 tile，沿时间轴一步完成）：**

```
b_gk = tl.cumsum(b_gate, axis=0)    # axis=0 = chunk 时间维 → [BT, BS] 前缀和
```

**尾 chunk 为何安全：** `tl.cumsum` 沿 axis=0 向后累加。3a 里把越界行清零，
补零只会出现在段尾，`0` 加到有效行的前缀和上不改变结果 —— 恰好等于真 kernel
`boundary_check` 返回 0 的行为，无需分支。

### 3c. log2 空间换算：`gk ← gk * scale`

**数学公式：**

```
gk[t,k] ← gk[t,k] * RCP_LN2          # RCP_LN2 = 1/ln2 = 1.4426950216293335
```

把 ln 空间的累计 gate 换到 log2 空间：下游 K4/K5/K6 全部用 `exp2(gk)`（而非 `exp`），
省一次换底，且硬件 exp2 比 exp 便宜 —— 这条乘法是刻意留在本 kernel 里的。

**逐 tile 分块代码：**

```
b_gk *= scale                         # 常量乘，仍在 [BT, BS] tile 上
store(gk + 同 tile 地址, b_gk)        # 写回 [B,T,H,K]
```

> 尾 chunk / 尾 S-tile 的越界列本就被清零，越界行在 3a 清零；写回同 mask，不产生脏数据。

## 4. 关键代码对应（`src/gate_kernel.py`）

- 内核：`_gate_cumsum_kernel`（`@triton.jit`）
  - program ID 与索引计算：`i_s, i_t, i_bh = tl.program_id(0..2)`；
  - tile 加载：行主序 flat 索引 `x + i_b*T*H*K + t*H*K + h*K + s`，mask 边界；
  - bias：`tl.load(dt_bias + i_h*K + tile_s)`（`HAS_BIAS` 编译期开关，none → 不读）；
  - 门控：`b_gate = -tl.exp(b_a) * _softplus_fwd(b_s)`；
  - cumsum：`b_o = tl.cumsum(b_gate, axis=0)`；
  - scale：`if HAS_SCALE: b_o *= scale`；
  - grid：`(cdiv(K, BS), cdiv(T, BT), B*H)`。
- 驱动：`gate_chunk_cumsum(...)`（NPU：triton；无 NPU 时自动退化为 `gate_cumsum_ref`）。
- CPU 参考：`gate_cumsum_ref(...)`（纯 torch，逐 chunk cumsum，任意 B/T/H/K）。

### 编译期特化路径

```
                HAS_BIAS?        HAS_SCALE?
                (T/F)            (T/F)
                   │                │
    +--------------┼────────+   +───┴───+
    |               │         |   load bias  不缩放
    |            不加载        |   (实际恒 T)  (恒 False)
    |               │         └───┬───┘
    |               └─────────────┘
    └────────────────────────────────┘
```

`HAS_BIAS` 与 `HAS_SCALE` 共 `2×2 = 4` 种组合，Triton 编译期为每条路径生成特化版本。

## 5. 数据流图

```
                       输入
                        |
       ┌───────────────┼────────────────┐
       |               |                |
       v               v                v
  raw_gate [B,T,H,K]  A_log [H]     dt_bias [H*K](可选)
       |               |                |
       |   ┌───────────┘                |
       |   |   ┌────────────────────────┘
       v   v   v
 ┌─────────────────────────────────────────────────────┐
 │ Grid: (cdiv(K,BS), NT, B*H)                          │
 │                                                     │
 │ CTA(i_s, i_t, i_bh):                                │
 │   - 时间维: chunk i_t, BT=64                        │
 │   - 通道维: S-tile i_s*BS..i_s*BS+BS                │
 │   - batch: i_b = i_bh // H                          │
 │   - head:  i_h = i_bh %  H                          │
 │                                                     │
 │  1. mask 加载 [BT,BS] raw gate tile, 加 bias         │
 │  2. gate = -exp(A_log) * softplus(x + bias)         │
 │  3. 无效行清零后 tl.cumsum(axis=0)                  │
 │  4. *= RCP_LN2; masked store                        │
 └─────────────────────────────────────────────────────┘
                        |
                        v
                  output [B,T,H,K] fp32
                  (激活 + cumsum + log2 空间)
```

### 跨 Chunk 边界示意

```
时间轴 T (T=256, BT=64, NT=4):
  ┌─────────┬─────────┬─────────┬─────────┐
  │ Chunk 0 │ Chunk 1 │ Chunk 2 │ Chunk 3 │
  │ t0..63  │ t64..127│128..191 │192..255 │
  └─────────┴─────────┴─────────┴─────────┘
     │         │         │         │
     各 chunk 内部独立 cumsum，互不依赖
   Chunk 0 输出: cumsum(gate[0:64])    ← 起始于 gate[0]
   Chunk 1 输出: cumsum(gate[64:128])  ← 起始于 gate[64]   (非 gate[63] 的累加值)
```

## 6. 精度 & 性能对比测试策略（配套 `run.py` / `testcases.csv`）

参考：`test_level2_kernel_precision.py::TestGateChunkCumsumKernel`

- 每个 case 固定 seed，输入分布与 level2 一致（`raw_gate = randn*0.5-2.0`、`A_log*0.1`、`dt_bias*0.1`）；
- 对每个 case 依次跑**两个对比方**：
  1. **torch_npu 元算子** `gate_cumsum_torch` —— 用 torch_npu 现成算子组合
     （`add`→`logaddexp` softplus→`exp`→`reshape`+`cumsum`→`mul`）完成同样计算，
     作为**精度基本准**和**性能基本准**；
  2. **triton kernel** `gate_chunk_cumsum` —— 本目录实现的单 kernel 版本。
- 精度指标（两个都满足才 PASS）：
  - `max|triton - torch_npu| < 1e-2`；
  - `max|torch_npu - CPU 参考| < 1e-2` 且 `max|triton - CPU 参考| < 1e-2`
    （CPU 参考 `gate_cumsum_ref` 为逐 chunk 循环的 ground truth）。
  实测 maxdiff 均为 fp32 累积噪声级（≤ ~1e-5）。
- 性能指标：预热 `--warmup`(默认 5) 次、各跑 `--repeats`(默认 30) 次，用
  `torch.npu.synchronize()` 包裹计时，输出每个 case 的
  **加速比 = torch_npu_time / triton_time**。
- CSV 中共 15 个 case，覆盖：完整 chunk / 尾 chunk 不满（`T=63/65/96/100/127/193/2562`）、
  单/多 head（`H=1/2/3`）、单/多 batch（`B=1/2`）、`K=32/64/128`
  （同时覆盖 `BS=32` 整数倍与非整数倍）。

> 实测（910B2, triton-ascend, 30 次计时）：15/15 PASS；所有 case 加速比 **1.15x–1.54x**
> （torch_npu 为多算子图多 kernel 分发，triton 单 kernel 内存访问更省）；
> 大 T（`tiny_T2562`）约 **1.42x**，K=32 最小（`k32_basic` 约 1.15x，算子图本身已很小）。

## 7. 性能测试思路（配套 `run.py`）

内存瓶颈型 kernel（读 `B*T*H*K`、写 `B*T*H*K`），性能对比的两个对象：

- **torch_npu 元算子**：`add + logaddexp(softplus) + exp + cumsum + mul` 的算子图，
  多次 kernel 启动 + 中间张量读写，是性能基准的下界参考；
- **triton kernel**：单 kernel 完成激活 + cumsum + 缩放，避免中间张量；
  每 case 预热 5 次、计时 30 次（`torch.npu.synchronize()` 包裹），报告
  `torch_ms / triton_ms / speedup`。

加速比主要来自:单 kernel 复用与 tunable 的 tile 选择(`BS=32`)——由于 gate 的激活
和 cumsum 都是内存带宽型,理论加速比接近「算子图启动/中间读写」的节省比例，
实测约 1.15x–1.54x。后续如做 kernel 级优化,对照变量：`BS`(32 vs 64)、`num_warps`、
是否用上 `exp2` 融合路径（配 `HAS_SCALE=constexpr`）等。

## 附录 A: 关键常量速查

| 常量 | 值 | 用途 |
|------|-----|------|
| `BS` | 32 | S 维 tile 大小 |
| `BT` | 64 | Chunk 大小 / 时间维 tile 大小 |
| `RCP_LN2` | 1.4426950216293335 | ln(2) 倒数, ln→log2 转换 |
| `SOFTPLUS_THRESHOLD` | 20.0 | softplus 线性近似阈值 |
| `num_warps` | 1 | 每 CTA warp 数（与真实 kernel 一致） |

## 附录 B: 与真实 kernel 的差异对照

| 项 | `python/.../fla/kda.py` | 本目录独立实现 |
|----|--------------------------|----------------|
| VARLEN/`cu_seqlens` | 支持 | 只做固定长度 |
| Safe gate / `lower_bound` | 支持 | 只做 Standard gate |
| `chunk_indices` 预计算 | 有 | 无 |
| 无 NPU 环境 | 无法运行 | `gate_cumsum_ref` 退化为纯 CPU 参考 |
| 依赖 | sglang 包 + `RCP_LN2` 等 | 仅 torch + triton |