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
    b_k = load k [i_t*BT:(i_t+1)*BT, :]         # 行主 [BT,K] block（整 K 单 dot）
    b_h += tl.dot(tl.trans(b_v), b_k)           # ④ 输入侧转置 [BV,BT]@[BT,K] → [BV,K]
                                                # 累加（数学等价 trans(dot(k,b_v))；
                                                # 后者在 CANN 9.1 编译失败，见 §5）
store initial_state[indices[b], h] 的 [i_v*BV:(i_v+1)*BV, :] = b_h   # 终态 in-place
```

> **h 快照与终态写回**：每个 chunk 一进就先 `h[i_t]=state`（输出给 K6 当该 chunk 的起始
> 记忆），跑完 NT 个 chunk 再把最终 state in-place 写回 `initial_state[indices[b],h]`
> （下个 batch 直接用）。triton-ascend 对 `(V,K)` 形状的 block_ptr store 有寄存器损坏 bug，
> 故快照与终态统一走行步长 K 的 2D 手动指针 `tl.store`（`_store_h_full`，任意 K 含 K=64；
> 勿改回 flat reshape —— CANN 9.1 的 expand_shape 会报错，见 §5）。
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

官方基线 ≈ **11.43 ms/调用**（±10%，2026-09-07 本机 triton-ascend 3.2.1 msprof 实测
口径；④ 输入侧转置形态，见 §5。旧 CANN 9.0 口径 9.1–9.5 仅作历史参考）；
门槛 `max_diff < 1e-2`。

## 5. 设计创新点与深度解析

> 本节复盘**基线 kernel 的设计决策链**（数字为迭代环境实测，量级参考；随工具链漂移）。

- 【串行性的归属设计】K5 是全链**唯一 chunk 间有依赖**的算子。state 逐 (b,h) 私有 ⇒
  让一个 CTA 独占一个 (b,h) 的整条 state 链（`grid=(cdiv(V,BV), B·H)`），NT=256 的
  串行递推在 CTA 内部跑完 —— 并行度交给 B·H×(V/BV) 个互不通信的 CTA。"把依赖留在
  块内、把并行留给块间"。
- 【BV=V 单调最优】BV 扫描 32/64/128 → 40/20/10ms：整 V 驻留一个 CTA 使每 chunk 恰好
  2 个 dot（② 预测 + ④ 外积），state 全程寄存器驻留；分块越多、每 chunk dot 数越多、
  寄存器-UB 往返越频繁 —— 扫描曲线直接给出最优，而非猜测。
- 【整 K 单 tile】K=128 时每 chunk 从 4 dot 降到 2 dot（9.78→9.09ms）；同一轮隔离
  实验测出本 kernel 的访存下界 ≈3.02ms —— "当前离下界还有多远"成为判断后续优化
  空间（而不是感觉"还能优化"）的锚点。
- 【软件流水线要接线】`tl.range(NT, num_stages=NS)` 是唯一能让预取生效的写法 ——
  早期直接传 `num_stages` 参数被 triton-ascend **静默忽略**（不报错、不生效）；
  NS=3 接线后 9.51→9.11ms。编译器行为必须实测，文档假设不可靠。
- 【衰减时点 = 语义设计】用 chunk 末 token 的累计 gk **一次**乘 state（③ 在②之后、
  ④ 之前）：② 用未衰减快照预测残差，③ 让旧记忆退场，④ 叠加的新记忆不背本次衰减
  —— 一个乘法的位置同时满足 delta-rule 语义与"64 token 只衰减一次"的性能需求。
- 【CANN 9.1 编译坑的根因级修复（本包内核形态）】`state += trans(dot(k, b_v))` 的
  "dot 输出转置 + 累加"链触发 hivm-plan-memory root-alloc 失败（误报 ub overflow）⇒
  改写为**输入侧转置** `dot(trans(b_v), b_k)`（行主 [BT,K] 加载、数学等价）。三形态
  实测 11.5 / 21.8 / 22.3ms，11.5ms 为当前最优可用（vs 旧工具链 9.15ms，+25% 是编译
  兼容代价，不是优化空间）；K=64 的 flat reshape store 触发 expand_shape bug
  （"collapsed dim size 2048 must equal 4096"）⇒ 快照/终态统一 2D 手动指针 store。
- 【收敛论证参考】NT=256 串行链 + cube 利用率仅 3–4%：除非把 delta-rule 更新写成
  可块化的数学形式（matrix-geometric / 半环前缀和类分解），chunk 链就是下界 ——
  注意 BT=128 能让链条减半，但破坏六算子共享的 chunk 契约（下游对不上），是无效捷径。
