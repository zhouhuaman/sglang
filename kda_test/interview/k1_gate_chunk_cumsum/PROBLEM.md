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

官方基线 ≈ **1.37 ms/调用**（±10%，2026-09-07 本机 triton-ascend 3.2.1 msprof 实测口径）；
门槛 `max_diff < 1e-2`。

## 5. 设计创新点与深度解析

> 本节复盘**基线 kernel 的设计决策链**（数字为迭代环境实测，量级参考；随工具链/机型
> 漂移）。K1 无矩阵乘，其优化主线与其余五个完全不同：纯访存形态 + chunk 前缀和指令。

- 【并行度建模】算子 = 逐元素激活 + 沿时间维的 **chunk 局部**前缀和。把三维并行度
  （通道块 × chunk × B·H）显式铺成 grid，而非把整段 [T,K] 交给一个大 kernel —— 前缀和
  在 chunk 内串行、跨 chunk 与跨通道全并行，切到 chunk 粒度才能与 K2..K6 共享同一套
  BT=64 边界。
- 【硬件约束识别：grid 展平上限】triton-ascend 把 3D grid 展平成 1D 后，总 CTA 数不能
  超过 65535（超限直接启动失败 `ERR00100`）。BS=32 时目标 case 展平 98304 超限 →
  通道维放大到 BS=128（K 不切），grid 降到 24576、每 CTA 密度翻倍（R2 ≈1.85ms）——
  "通道怎么切"首先是被 CTA 上限逼出来的，其次才是 UB 容量。
- 【尾 chunk 补零安全性论证（免掩码的关键）】`tl.cumsum` 沿时间轴**单向**累加 ⇒ 越界行
  清零后，补零只出现在段尾、不会污染头部有效行的前缀和 ⇒ 整个 kernel 不需要
  boundary_check 的 if 分支、store 不加 mask。这是"用数学性质换代码形态"的典型，
  与 K2 的 exp2 拆分、K3 的截断逆同族。
- 【跨算子协同：log2 空间换算】把 `scale = RCP_LN2` 刻意放在本 kernel 尾乘，将 gate 从
  ln 空间换算到 log2 空间 —— 下游 K2/K3/K6 的 `exp2(g[i]-g[j])` 才能直接当底，硬件
  exp2 比 exp 便宜，K4/K5 的 per-channel 衰减也省一次换底。一个常量放对位置 = 全链
  省一类指令。
- 【UB 预算实测】BT=128×BS=128 的 fp32 tile 直接报 `ub overflow: 2625536 bits >
  1572864 bits` —— 用编译器报错反推出片上 UB 可用预算，"尽量放大 tile"因此有了硬上界
  （128×128 fp32 = 64KB 级别即撞 UB），tile 形状不是拍脑袋选的。
- 【数值契约】`gate = −exp(A_log)·softplus(x+bias)` 恒负 ⇒ `gk` 单调递减（衰减随 token
  加深），这是下游语义的一部分；softplus 在 v≥20 退化为 v 防溢出 —— 数值稳定性解决在
  公式层，kernel 层只负责把它算对。
