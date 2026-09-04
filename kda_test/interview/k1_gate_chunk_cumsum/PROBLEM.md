# 问题 K1 · gate_chunk_cumsum

> 目录文件：`gate_chunk_cumsum_kernel.py` 里有 torch 参考 `gate_chunk_cumsum_torch` 与
> 待改写/待优化的 triton kernel `gate_chunk_cumsum_triton`（当前为收敛版）。你只改
> kernel；`test.py` 负责比对与计时。

## 0. 算子描述

把每个 token 的"逐通道门控值"过一层激活，再做 **chunk 内**的前缀和，输出 log2 空间的
累积衰减 `gk`（供后续 exp2 系列 kernel 使用）。衰减随 token 数线性加深：同一个 (b,h,通道)
在 64-token chunk 内，第 k 个 token 的衰减 = chunk 内前 k 个 token 的门控之和。跨 chunk
**不传递**（每个 chunk 从 0 重新开始）。

## 1. 输入

| 张量 | shape | 含义 |
|---|---|---|
| `x` | [B, T, H, K] | 每个 token 逐通道的原始门控（未激活） |
| `A_log` | [H] | 每 head 一个对数尺度标量，控制整 head 的衰减强度 |
| `dt_bias` | [H*K] | 每 head 每通道的偏置，平铺 [H*K]；`dt_bias[h*K+k]` 属于 (h,k) |
| `chunk_size` | int | 常数 64（不许改） |
| `scale` | float | 常数 `RCP_LN2 ≈ 1.4427`（=1/ln2，把结果从 ln 空间换算到 log2 空间） |

维度：`B`=batch；`T`=token 序号；`H`=head；`K`=通道（q/k/gk 最后一维，本算子 K=128）。
`NT = T/64`。

## 2. 输出

| 张量 | shape | 含义 |
|---|---|---|
| `gk` | [B, T, H, K] | 激活 + chunk 内前缀和后的累积门控（log2 空间） |

## 3. 计算（先划 tile → 单 tile 内全部计算）

> 记号：`BT=chunk_size=64`；`softplus(v)=log(1+e^v)`（v≥20 时≈v，防溢出）；`exp` 是
> 自然指数。前缀和沿**时间轴**、逐通道独立 —— 本算子是一次"激活 + 按 64 步一段做
> prefix-sum + log2 缩放"的 elementwise 变换，无矩阵乘。

### 划 tile：切成 B·H·NT 个 tile

输出 `gk[B,T,H,K]` 在 (batch, head, chunk, 通道) 上全并行。收敛 kernel 每个 CTA 处理
一个 **(batch,head) 的一个 chunk × 整条 K 通道**：K 维不切（`BS=128 ≥ K` ⇒
`cdiv(K,BS)=1`），tile 形状 `[BT,K]`。

```
grid = (cdiv(K,BS), NT, B*H)      # 目标 K=128、BS=128 → (1, NT, B*H)，共 B·H·NT 个 tile
program_id: (0)=通道块(恒 0), (1)=chunk, (2)=联合 (batch,head)
一个 tile: x[tc..tc+63, 0..K-1]；tc = chunk 首 token = i_t*BT
```

### 单 tile 公式（tile = (b,h) 的一个 chunk 块 `[BT,K]`）

载入该 chunk 的原始门控 `x_c` 与偏置，先逐元素激活、再沿时间轴（行）前缀和：

```
gate[BT,K] = -exp(A_log[h]) · softplus( x_c[BT,K] + dt_bias[h,:] )   # ① 激活
gk[BT,K]   = RCP_LN2 · cumsum_行( gate[BT,K] )                        # ② 前缀和 × log2
```

- `dt_bias[h,:]` 按行广播：同一 (h,通道) 所有 token 共用一个偏置；`softplus` 保证括号内
  为正，`A_log[h]` 每 head 一个标量、控制整 head 衰减强度；**负号** ⇒ `gate` 恒负 ⇒
  `gk` 单调递减（衰减随 token 数加深）。
- `RCP_LN2≈1.4427` 把 ln 空间换算到 log2 空间。刻意留到本 kernel 尾乘 —— 下游 K4/K5/K6
  全用 `exp2(gk)`（log2 空间的指数直接当底），省一次换底；硬件 `exp2` 也比 `exp` 便宜。

### tile 代码（= kernel 主体，注释即全部逻辑）

```
i_s, i_t, i_bh = tl.program_id(0..2)      # (通道块, chunk, 联合 (b,h))；b=i_bh//H, h=i_bh%H
tc = i_t*BT ;  s0 = i_s*BS
rows = tc + tl.arange(0, BT) ;  cols = s0 + tl.arange(0, BS)
mask = (rows[:,None] < T) & (cols[None,:] < K)          # 越界 load 为 0
ptr_x = x + b*T*H*K + h*K + rows[:,None]*(H*K) + cols[None,:]
b_x = tl.load(ptr_x, mask=mask, other=0.0).to(tl.float32)   # [BT,BS] 原始门控
b_b = tl.load(dt_bias + h*K + cols, mask=cols<K, other=0.0).to(tl.float32)
b_x = b_x + b_b[None,:]                                   # ① 加偏置，按列广播
b_gate = -tl.exp(tl.load(A_log + h)) * _softplus(b_x)     # ② 激活（A_log 每 head 标量）
b_gate = tl.where(mask, b_gate, 0.0)      # 无效行清零（cumsum 前，tail chunk 安全）
b_gk = tl.cumsum(b_gate, axis=0) * scale  # ③ chunk 内前缀和(axis=0=时间维) × log2
ptr_o = o + b*T*H*K + h*K + rows[:,None]*(H*K) + cols[None,:]
tl.store(ptr_o, b_gk)                     # 写回 gk [B,T,H,K]
```

> 尾 chunk 为何安全：`tl.cumsum` 沿 axis=0 向后累加，无效行清零后补零只出现在段尾，不改变
> 头部有效行的前缀和 —— 等价于真 kernel `boundary_check` 返回 0 的行为。

## 4. 约束与验收

- 评分 case：`B=1, T=16384, H=96, K=128`（NT=256，T%64==0）；输入输出 fp32。
- 正确性：与 `gate_chunk_cumsum_torch` 一致，逐元素差 < `1e-2`（当前 ~6e-5）。
- **不许改默认 `chunk_size=64`**（破坏上游契约 = 无效解）。`scale`、`A_log`、`dt_bias`
  视为给定常量/输入，不可绕开（改语义/硬编码某 case 值 = 无效解）。

```bash
source ../env.sh                        # 容器环境（可 `source ../env.sh 4` 固定到空闲卡）
python3 test.py                         # ① 正确性：PASS + max_diff
msprof --output=./prof_k1 --application="python3 test.py --perf --repeats 7 --warmup 3" \
    && python3 test.py --report ./prof_k1        # ② 基线（msprof Task Duration 每调用均值）
```

官方基线 ≈ **1.96 ms/调用**（±10%）；门槛 `max_diff < 1e-2`。
