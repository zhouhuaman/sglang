# 问题 K5 · delta_rule_h

> 目录文件：`delta_rule_h_kernel.py` 里有 torch 参考 `delta_rule_h_torch` 与待改写/待优化
> 的 triton kernel `delta_rule_h_triton`（当前为多轮收敛版）。你只改 kernel；`test.py`
> 负责比对与计时。

## 0. 算子描述

用一张会在线更新的小矩阵记忆 `state`（[V,K]）逐 chunk 串行扫过序列。每个 64-token chunk：
① 把 `state` 快照写盘（输出 `h`）；② 用 `state` 预测每个 token 的 value、与真实 `u` 比出
残差（输出 `v_new`）；③ `state` 按 gate 衰减一次；④ 把本 chunk 残差 × key 的外积累加进
`state`。终态 in-place 写回 `initial_state`。只有 chunk 顺序是串行依赖。

## 1. 输入

| 张量 | shape | 含义 |
|---|---|---|
| `k`（也称 kg） | [B, T, H, K] | 每个 token 的（衰减后）key，用于写记忆 |
| `w` | [B, T, H, K] | 每个 token 的预测权重，用于从记忆读预测 |
| `u` | [B, T, H, V] | 每个 token 的真实 value（预测目标） |
| `gk` | [B, T, H, K] | 逐通道 gate（log2 空间，**已累计**）；每 chunk 用末尾 token 一次衰减 |
| `initial_state` | [N, H, V, K] | 初始记忆池；**本算子 in-place 更新为终态** |
| `indices` | [B] | batch `b` 用池中第 `indices[b]` 行 |

维度：`B`=batch；`T`=token 序号；`H`=head；`K`=key/通道长度；`V`=value 特征通道
（约束 **`K == V`**）；`N`=记忆池行数（test 里 `N==B`、`indices=arange(B)`）。`NT = T/64`。

## 2. 输出

| 张量 | shape | 含义 |
|---|---|---|
| `h` | [B, NT, H, V, K] | 每个 (batch, chunk, head) 在**预测前**的 `state` 快照 |
| `v_new` | [B, T, H, V] | 每个 token 的残差 |
| `initial_state` | [N, H, V, K] | （in-place）终态 |

## 3. 计算（先划 tile → 单 tile 内全部计算）

> 记号：`BT=64`；`state[V,K]` 是 (b,h) 私有的记忆矩阵：第 `v` 行 = "把 key 特征映到
> value 第 `v` 维"的线性系数。`@`=矩阵乘；`ᵀ`=转置；`⊙`=逐元素乘（按行广播）；
> `exp2(v)=2**v`。一个 chunk 内四步：前两步用**进 chunk 时未衰减**的 state，后两步是
> "旧记忆退场 + 写入本 chunk 新记忆"。目标 `K=V=128` → state 整块 [128,128] 驻留一个 CTA。

### 划 tile：每 CTA = 一个 (b,h) 的整条 state；chunk 链在 tile 内串行

`state` 是逐 (b,h) 私有的，唯一的串行依赖是 **chunk 顺序**，而它正好发生在一个 CTA
内部 —— 收敛 kernel 把每个 (batch, head) 的 state（的 BV 行 slab）分给一个 CTA，让它在
**自己里面**把 NT 个 chunk 递推跑完：

```
grid = (cdiv(V,BV), B*H)      # 目标 BV=V → (1, B·H)；tile = (b,h) 的 state [BV,K] slab
program_id(0)=V-slab i_v, (1)=联合 (b,h)；一个 tile: 逐 i_t=0..NT-1 串行四步，快照写 h
```

### 单 tile 公式（tile 内的 chunk 递推；state 即该 (b,h) 的私有寄存器）

对 chunk i_t（行区间 `i_t*BT .. i_t*BT+63`），用**进 chunk 时未衰减**的 state 算前两步：

```
h[i_t]      = state                            # ① 快照（供 K6 当本 chunk 起始记忆）
v_new[i_t]  = u[i_t] − W[i_t] @ stateᵀ          # ② 残差：state 预测 vs 真实 u
state      *= exp2(gk_last[i_t])[None,:]       # ③ 遗忘：整 chunk 一次（末 token gate）
state      += v_new[i_t]ᵀ @ k[i_t]             # ④ 写记忆：残差外积累加
```

- ② `pred = W @ stateᵀ`：state 第 v 行当系数、读记忆预测每个 token 的 value；残差 = 真实 u
  − 预测。整块 `[BT,K] @ [K,BV] → [BT,BV]`。
- ③ `gk_last = gk[chunk 末有效 token, :]`（[K]）：gk 是 chunk 内**已累计**的 log2 gate，
  故 64 个 token 不必逐 token 乘，整 chunk 只在边界乘一次、用末 token 的累计值 —— 正好把
  state 从"进 chunk"推进到"出 chunk"时刻。
- **为何 ③ 在 ② 之后、④ 之前**：② 用未衰减快照预测残差（预测才反映"此刻该记得什么"）；
  ③ 让**旧**记忆按 gate 退场，本 chunk 新写的残差在退场**之后**才叠加（④），故新记忆不背
  本次衰减 —— 这正是 delta rule"本 chunk 修正"的语义。
- ④ 是外积累加 `state += v_newᵀ @ k`，整块 `[V,BT]@[BT,K] → [V,K]`。

### tile 代码（= kernel 主体，注释即全部逻辑）

```
i_v, i_nh = tl.program_id(0..1)                # (V-slab, 联合 (b,h))；解码 b,h
b_h = load initial_state[indices[b], h] 的 [i_v*BV:(i_v+1)*BV, :]    # 初始记忆 [BV,K]
for i_t in tl.range(NT, num_stages=NS):        # chunk 链（唯一串行维，软件流水预取）
    store h[i_t] = b_h                          # ① 快照（进 chunk 时，未衰减）
    b_w = load w  [i_t*BT:(i_t+1)*BT, :]                # [BT,K]
    b_pred = tl.dot(b_w, tl.trans(b_h))                 # [BT,K]@[K,BV] → [BT,BV]
    b_u = load u [i_t*BT:(i_t+1)*BT, i_v*BV:(i_v+1)*BV] # [BT,BV]
    b_v = b_u − b_pred                                   # ② 残差
    store v_new[i_t] = b_v                      # 写残差（供 K6）
    last = min((i_t+1)*BT, T) − 1               # chunk 末有效 token
    b_gn = load gk [last, offs_k]               # [K] 逐通道累计 gate
    b_h *= exp2(b_gn)[None, :]                  # ③ [1,K] 广播到 [BV,K] 逐列乘
    b_k = load k [i_t*BT:(i_t+1)*BT, :]  #（实载为转置 block [K,BT]）
    b_h += tl.trans(tl.dot(b_k, b_v))           # ④ [K,BT]@[BT,BV] →ᵀ→ [BV,K] 累加
store initial_state[indices[b], h] 的 [i_v*BV:(i_v+1)*BV, :] = b_h   # 终态 in-place
```

> **h 快照与终态写回**：每个 chunk 一进就先 `h[i_t]=state`（输出给 K6 当该 chunk 的起始
> 记忆），跑完 NT 个 chunk 再把最终 state in-place 写回 `initial_state[indices[b],h]`
> （下个 batch 直接用）。triton-ascend 对 `(V,K)` 形状的 block_ptr store 有寄存器损坏 bug，
> 故 K=64 走 flat 1D store、K≠64 走通用 2D 手动指针 store（勿改回 flat reshape）。
>
> 尾 chunk 不满 BT 时 `last=min(…,T)−1` 取到末有效 token、越界行补 0 即可（参考同样补 0
> 再裁）。`h`、`v_new`、终态三者都要与参考一致（§4 口径）。

## 4. 约束与验收

- 评分 case：`B=1, T=16384, H=96, K=V=128`（NT=256，T%64==0）；输入输出 fp32。
- 正确性：`h`、`v_new`、终态三者和 `delta_rule_h_torch` 一致，逐元素差 < `1e-2`
  （当前逐位一致，max_diff=0）。
- **不许改默认 `chunk_size=64`**（改大如 128 会破坏上游契约 = 无效解）。递推跨 chunk
  串行，"并行化 chunk 链"的方案改变语义、视为无效。
- `initial_state` 两版都 in-place 写回：`test.py` 每次调用前会 `clone` 避免互相污染，
  你自行 bench 也要 clone。

```bash
source ../env.sh                        # 容器环境（可 `source ../env.sh 4` 固定到空闲卡）
python3 test.py                         # ① 正确性：PASS + max_diff（torch 参考 96 head × 256 chunk 串行 matmul，约 10–20s，请耐心）
msprof --output=./prof_k5 --application="python3 test.py --perf --repeats 7 --warmup 3" \
    && python3 test.py --report ./prof_k5        # ② 基线（msprof Task Duration 每调用均值）
```

官方基线 ≈ **9.1–9.5 ms/调用**（±10%）；门槛 `max_diff < 1e-2`。
