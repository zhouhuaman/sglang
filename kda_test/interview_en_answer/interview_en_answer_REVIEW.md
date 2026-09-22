# KDA 面试包复验报告 · `interview_en_answer/`

> **内部评审文档，勿进交付包、勿给候选人。**
> 复验对象：`kda_test/interview_en_answer/`（候选人提交：k1/k2/k6 三题 kernel + `REPORT.md`）。
> 对照物：`kda_test/interview/`（出厂 baseline，未改动）。
> 复验日期：2026-09-22。复验人：zhm 侧。

---

## 0. 结论速览

| 项 | 结论 |
|---|---|
| 目标 case 正确性（评分门槛） | 三题全 PASS，余量 4~6 个数量级；未改动测试脚手架 |
| K1 gate_chunk_cumsum | ✅ 真实增益，受控复测 **1.29×**（报告声称 1.37×） |
| K2 token_parallel | ⚠️ 官方口径 ≈1.00×（报告声称 1.08×）；但**算子 wall-clock −19.4%**，收益真实存在、只是落在指标盲区 |
| K6 gla_output | ✅ 复现，**1.034×**（报告声称 1.04×），两轮稳定 |
| 综合评分 | **66 / 100（C+，及格偏上，建议复面）**；若改按算子 wall-clock 计分则为 **75 / 100** |
| 最大风险 | K2 的 baseline 数据不可复现（报告 9.466 vs 实测 8.683，重复性 0.01%），且优化前后未在同一条件下测量 |

**一句话画像**：对 Ascend/Triton-Ascend 有真实手感、能做指令级 profile、能做数学重写，也敢于用实测否掉自己的方案；但三题深度极不均衡，只有 K1 被做透，K2/K6 更像是同一套话术的复述，且**缺了最关键的一环——自证**。

---

## 1. 复验方法与环境

- 容器 `triton-ascend-env-zhm`（Triton-Ascend 3.2.1 / CANN 9.0.0 / torch_npu 2.7.1 / Python 3.11），8×910B2。
- **空闲卡 card 3**（card 4~7 处于 56GB 占用状态，已避开；开工前后各 `npu-smi info` 确认）。
- 协议与题目一致：`msprof --output=<dir> --application="python3 test.py --perf --repeats 7 --warmup 3"`，取 `op_summary_*.csv` 的 `Task Duration(us)` **每调用均值**（calls=10，含 warmup）。
- **baseline 与 candidate 成对、背靠背、同一 session** 测量，共 **两轮独立复测**（P1/P2）。
- 复验脚本（宿主侧，未跟踪）：
  - `kda_test/_verify_answer_inner.sh` —— 目标 case 正确性 + 两口径性能，base/cand 成对
  - `kda_test/_verify_answer_extra_inner.sh` —— 边界形状 `--selftest` + 第二轮性能复测
  - 容器内结果目录 `/tmp/kda_verify`、`/tmp/kda_verify2`

### 测量噪声（两轮之差）

| | baseline P1 / P2 | candidate P1 / P2 |
|---|---|---|
| K1 kernel | 1.840 / 1.846 ms（0.3%） | 1.428 / 1.428 ms（0.0%） |
| K2 kernel | 8.683 / 8.682 ms（**0.01%**） | 8.663 / 8.567 ms（1.1%） |
| K6 kernel | 4.524 / 4.546 ms（0.5%） | 4.382 / 4.380 ms（0.05%） |

→ 本机 kernel 口径重复性 **≤1%**，baseline 尤其稳定（K2 两轮只差 1.4us）。这个前提很重要：后面"s"无法复现"的结论不是机器噪声导致的。

---

## 2. 目标 case 实测（B=1, T=16384, H=96, K=V=128）

### 2.1 官方口径（op_summary 内 triton kernel `Task Duration` 每调用均值）

| 题 | baseline P1 / P2 | candidate P1 / P2 | **复验加速** | 报告声称 | 判定 |
|---|---|---|---|---|---|
| K1 | 1.840 / 1.846 ms | 1.428 / 1.428 ms | **1.29× (−22.4%)** | 1.37× (−27.0%) | ✅ 真增益，幅度略夸大 |
| K2 | 8.683 / 8.682 ms | 8.663 / 8.567 ms | **1.005× (−0.5%)** | 1.08× (−7.6%) | ❌ 未复现 |
| K6 | 4.524 / 4.546 ms | 4.382 / 4.380 ms | **1.034× (−3.3%)** | 1.04× (−3.7%) | ✅ 复现 |

官方隔离基线：K1 ≈1.96、K2 ≈9.5–9.8、K6 ≈4.53 ms/调用。K1（实测 1.840）偏快 6%、K6（4.524）吻合、**K2（实测 8.683）低于官方下沿 9.5 约 8.6%**——见 §4。

### 2.2 算子口径（`test.py --perf` wall-clock，含 driver 侧 torch 算子；非官方权威值）

| 题 | baseline P1 / P2 | candidate P1 / P2 | **算子加速** | 官方口径加速 |
|---|---|---|---|---|
| K1 | 2.097 / 2.082 ms | 1.678 / 1.703 ms | **−19.3%** | −22.4% |
| **K2** | **11.096 / 11.110 ms** | **8.987 / 8.882 ms** | **−19.4%** | −0.5% |
| K6 | 4.879 / 4.880 ms | 4.718 / 4.720 ms | −3.3% | −3.3% |

**这是本次复验最重要的发现**：K2 的收益不在 kernel 里，而在 kernel 之外（§5.2）。

---

## 3. 正确性

### 3.1 目标 case（评分门槛 `max_diff < 1e-2`）

| 题 | baseline | candidate | 判定 |
|---|---|---|---|
| K1 | 6.104e-05 | 6.104e-05 | PASS |
| K2 | 1.788e-07 | 1.490e-07 | PASS |
| K6 | 7.451e-08 | 7.451e-08 | PASS |

K1/K6 两者 max_diff 完全相同，说明误差主要由 torch 参考实现自身的精度决定，kernel 侧已到 fp32 极限。

**脚手架未被改动**：`test.py` 与出厂版做了 AST 级比对，差异仅限 docstring/注释/print 文案（中文→英文）；门槛常量 `MAX_DIFF=1e-2`、`TARGET`、`TRITON_NAME` 过滤器、`seed=20260825`、msprof 包裹命令全部保持原样。**这点是干净的。**

### 3.2 边界形状 `--selftest`（次要看点，非评分门槛）

| 题 | baseline | candidate |
|---|---|---|
| K1 | **7/7 PASS** | **7/7 PASS**（含尾块 T=63/127/193/2562、K=32、H=8、B=2） |
| K2 | FAIL（首个 case 即崩：`Akk.copy_` 63 vs 64 广播失败） | FAIL（同类：返回未截断的 Aqk，63 vs 64） |
| K6 | **7/7 PASS** | **1/7 PASS，6/7 FAIL**（max_diff 0.27~0.73，量级=未写入的垃圾值） |

- K1：**保留了完整的边界通用性**，报告里也明确写了"目标 case 用不上但仍保留边界检查"——言行一致。
- K2：属于**出厂 baseline 就有的既有缺陷**（尾块 padding 语义），不算候选人引入的回归；但其新 kernel 静态上还丢了 batch 维（见 §5.2）。
- K6：**这是明确的回归**。baseline 7/7 全过，candidate 除 `kv_mix`（V=128、T%64==0、B=1）外全错。根因：非 HM 路径把 `NT = tl.cdiv(T, BT)` 改成 `NT = T // BT`（尾块直接归零）、`v_tiles = V // BV`（V<128 时为 0，整个 grid 空转不写输出）、HM 路径硬编码 `bos = 0`（B>1 错）。**即该 kernel 只在被调优的那一个形状上正确。**

---

## 4. 报告声称 vs 实测（逐项对照）

| 报告声称 | 实测 | 结论 |
|---|---|---|
| K1 baseline 1.999 ms | 1.840 / 1.846 ms | ❌ 偏高 8.6% |
| K1 baseline→1.459 ms（1.37×） | 1.428 / 1.428 ms | ✅ 优化值吻合，增益应为 1.29× |
| K1 中间值 1.622 / 1.579 ms | 无中间版本可测 | ⚠️ 无法验证 |
| K2 baseline 9.466 ms | 8.683 / 8.682 ms（0.01% 重复性） | ❌ **不可复现** |
| K2 →8.742 ms（1.08×） | 8.663 / 8.567 ms | 优化值近似；但其声称的"baseline 9.466"下真实加速仅 ~0.5%，且 8.742 **慢于实测 baseline 8.683** |
| K6 baseline 4.522 ms | 4.524 / 4.546 ms | ✅ 吻合 |
| K6 →4.355 ms（1.04×） | 4.382 / 4.380 ms | ✅ 复现（1.034×） |
| cube 版 cumsum 4.603 ms、cube time 309.6us、vector cumsum ≈125us、指令级表格 | 无对应代码可测 | ⚠️ 方法可信、数值未验证 |

**关于 K1/K2 baseline 偏高的解读**：两题 baseline 都偏高约 8~9%，而优化值都与我实测吻合——符合"baseline 在受扰/不同条件下测、优化值在安静条件下测"的特征。K1 因为真实增益大，扣除后仍有 1.29×；K2 的所谓增益则被完全解释掉。K6 的 baseline 是准的，故 K6 的增益成立。

> 注：题目 README 明确要求"baseline 要自己复现、不要跨机器/跨时间窗强行比较、每次跑前用 `npu-smi info` 确认空闲卡"。K2 这一项未达标。

---

## 5. 逐题代码分析

### 5.1 K1 — 质量最高的一题

**改了什么**

1. grid 由 3D `(_cdiv(K,BS), num_chunks, B*H)` 拍平为一维 `(48,)`（48 = vector core 数）；`per_core = ceil(total_work/48)`，每核处理**连续**区间 `range(pid*per_core, min(total, (pid+1)*per_core))`（原为 `range(pid, total, num_programs)` 的交错分配）。
2. `multibuffer=False` —— 直面前述编译失败。
3. 显式 mask 计算 `masks = (tile_t[:,None] < T) & (tile_s[None,:] < K)` 全部换成 `tl.make_block_ptr(boundary_check=(0,1), padding_option='zero')`；删除已成冗余的 `tl.where(masks, b_gate, 0.0)`。
4. 运行期标量 `T` 改为 `tl.constexpr`（去掉 `do_not_specialize=['T']`）。仍保留 `tl.cumsum`。

**亮点**

- **编译失败定位**：报错 `ub overflow, requires 1581312 bits while 1572864 bits available` + `Failed to run BiShengHIR pipeline`。他提出"编译器对 work 循环做了多缓冲，导致多轮迭代 buffer 同时存活"的假设并用 `multibuffer=False` 化解——代码中该行确实存在，且最终收益成立，说明理解 UB 预算（192KB）与多缓冲的相互作用，不是瞎试。
- **执行域对齐核数**：知道 cumsum 在 vector 侧 → 用 48 而不是 cube 的 24。
- **标量寻址成本意识**：知道该后端上整数比较/地址向量生成可能退回标量，故用 block pointer 的 boundary check 替代显式掩码——与这台机器"标量寻址受限"的实际瓶颈模型吻合。
- **边界通用性未被牺牲**：7/7 边界形状通过。
- **放弃方向有真论证**：用指令级 profile 证明 `tl.cumsum` 被降级为 9,999 次迭代的标量串行链，再依次否掉 cube（矩阵乘形式，4.603ms）、Hillis-Steele 6 级 `tl.gather` 并行扫描（UB 超限），并给出"prefix sum 不是主瓶颈，kernel 仍以访存为主"的收敛结论。

### 5.2 K2 — 方案是真的，指标是瞎的

**改了什么**

1. 新增 `_token_parallel_kernel_new`，grid `(24,)`（24 = cube core 数）；`PER_CORE = H*NT // 24`，`tl.range(begin, end, num_stages=2)`，work_id 按 head-major 解码（`h = work_id // NT; c = work_id % NT`），即相邻迭代走同一 head 的相邻 chunk。
2. 每 chunk 两个大 dot（`Qeg@Keᵀ`、`Keg@Keᵀ`），只保留 4 个对角 16×16 下三角块（`same_subchunk & causal`），Aqk 无掩码连续写回。
3. **kernel 内 `tl.gather(Akk_full, col_idx, axis=1)` 收拢对角块**，直接写紧凑 `[B,TP,H,BC]` 的 Akk —— 省掉满宽 scratch（402MB）写 + driver 侧 `torch.gather`。
4. `return Aqk, Akk` 之后保留了约 60 行**不可达**原 dispatch（head-merge 路径、K 非 2 幂回退）。

**为什么官方口径看不到收益**

baseline 的 kernel 只占 8.68ms，但算子整体 11.10ms —— **中间 2.4ms 全在 kernel 外**：每次调用的满宽 scratch + driver 侧 `Akk.copy_(torch.gather(...))`（出厂代码注释自述"~2ms/调用"）。而官方指标是 `op_summary` 中按 `_token_parallel_kernel` 过滤的 `Task Duration`，**根本不统计 torch 侧的 Gather**。候选人把这块几乎删干净（算子侧 8.99ms，driver 开销降到 ~0.3ms），官方口径自然毫无反应。

**思路里可迁移的部分**

- **把 epilogue 收进 kernel**（最有价值的一条）。注意出厂代码里**专门有注释说这条路走不通**："triton-ascend 3.2.1 上 `tl.gather` 的 src 为 `tl.dot` 输出时结果错误（实测 Akk max_diff≈0.32）且退化 ~3× 慢"，因此被回退为 hm2。候选人换了排布重做，结果正确（1.49e-07）且不慢——**等于把代码库已放弃的设计捡了回来**，说明该后端限制是"排布相关"而非绝对。这条信息对后续优化本算子可直接复用。
- grid 对齐 cube 核数 + `tl.range(num_stages=2)` 流水：本题实测几乎为 0（kernel 8.66 vs 8.68），方案中性。
- head-major 连续排布：本题 dot-bound，中性。

**问题**

- **无消融**：24/1536 等不同 grid 没有对比；无法判断哪一步有效。
- **无 baseline 回测**：这是"官方口径 0.5% vs 报告 7.6%"的直接来源——他并不知道自己赢在算子侧，把功劳错记在 kernel 上。
- **代码卫生**：`PER_CORE = WORK // num_programs` 用截断除法（不能整除时**静默丢工作**）；base 指针无 batch 项（`q + h*K`），而 baseline 有 `bos = i_b*T`，**batch 维被丢失**；`return` 之后约 60 行死代码；docstring 仍描述已被绕过的旧路径。

### 5.3 K6 — 小增益，代价是通用性

**改了什么**：两个 kernel 都改成一维 `grid=(24,)` + `per_core` 连续区间；全面改用 block pointer；HM 路径带 `boundary_check`，非 HM 路径直接 `tl.load/tl.store` 无掩码；HM 路径硬编码 `bos = 0`；非 HM 路径 `NT = T // BT`。删除了原按 `(_cdiv(V,BV), NT, B*H)` 计算的 grid。

**实测**：1.034×，与声称的 1.04× 吻合，两轮稳定。**但**该 kernel 只在 `V=128、T%64==0、B=1` 时正确（§3.2），本质是"针对目标形状特化"，用通用性换了 3% 的收益。

---

## 6. 能力画像

**展现出来的能力**

1. **编译/后端失败定位**（K1，含金量最高）：从 BiShengHIR 报错反推到多缓冲，并用 `multibuffer=False` 验证假设。
2. **硬件拓扑 → 启动配置映射**：按执行域对齐核数（cumsum→48 vector、dot→24 cube），一维拍平 + 连续区间分配。
3. **访存与标量寻址成本意识**：block pointer 替代显式掩码、减少整数比较与地址向量生成、提升局部性。
4. **profiling 驱动的决策闭环**：能用最小复现 kernel 做隔离、看指令级行为，并**用实测否掉自己的方案**（cube 版、并行扫描），这是整包里最强的信号。
5. **数学重写**：前缀和→下三角矩阵乘；识别对角块结构；kernel 内 `tl.gather` 收拢。
6. **重构保正确性**：K1 拍平后 7/7 边界形状仍正确。
7. **表达结构化**：假设—改动—前后数字，外加真实的"放弃方向"一节；负结果不藏。

**欠缺的能力**

1. **测量纪律（最致命）**：优化前后未在同一条件下测；K2 的 baseline 数据不可复现；三题里只有 K6 的 baseline 是准的。
2. **区分真实收益与噪声**：报 7.6% 实际 0.5%；全文无重复测量/方差估计。
3. **跨题深度迁移**：K2/K6 只是 K1 技巧的复述，无 op_summary、无消融、"predominantly memory-bound"是断言而非论证。
4. **代码卫生与不破坏既有设计**：死代码、丢 batch、截断除法、硬编码核数、docstring 失真；对出厂代码里已注明的后端坑缺乏呼应。
5. **可移植性**：`grid=(48,)/(24,)` 硬编码，换核数的机器直接失效。

---

## 7. 评分明细

### 按官方口径（op_summary kernel Task Duration）

| 维度 | 权重 | 得分 | 依据 |
|---|---:|---:|---|
| 目标 case 正确性（评分门槛） | 15 | 15 | 三题 PASS，余量大，未动脚手架 |
| K1 优化 | 25 | 21 | 复现 1.29×；诊断扎实；边界 7/7 仍正确；唯一扣分是 baseline 虚高导致声称 1.37× |
| K2 优化 | 20 | 4 | 官方口径 1.005×；baseline 不可复现；死代码 + 丢 batch + 无消融 |
| K6 优化 | 15 | 12 | 复现 1.034×，两轮稳定；但只在整除形状正确 |
| 瓶颈诊断与证据 | 15 | 9 | K1 有编译日志/指令级 profile/隔离实验；K2/K6 只有断言 |
| 报告与测量纪律 | 10 | 5 | 结构清晰、"放弃方向"是真亮点；但头号数字经不起复测，违反包内明写的测量纪律 |
| **合计** | **100** | **66** | **C+，及格偏上** |

### 口径敏感性

若按**算子 wall-clock** 计分（即 K2 的 epilogue 优化能被计入），K2 一格应从 4 → 13，总分 **75/100（B−）**。

关键区别在于：**"没做出来"与"做出来了但指标看不见"是两种性质**。本复验判定 K2 属于后者——他的方案是对的、收益是真的（−19.4%），失分应落在"没做 baseline 回测、没做消融、留下死代码与丢 batch"，而不是"没有收益"。

---

## 8. 对面试包本身的建议

1. **K2 的官方指标存在结构性盲区**：`op_summary` 里按 triton kernel 名过滤的 `Task Duration` 不统计 driver 侧 torch 算子，而 baseline 明摆着留了 ~2ms/调用的 `torch.gather` + 满宽 scratch 给候选人删——**按现指标删了也是 0 分**，等于系统性惩罚该题唯一正确的优化方向。建议二选一：改按算子 wall-clock 计分，或在 `PROBLEM.md` 中明确"kernel 外 epilogue 不计入本次评分"。
2. **K6 的 baseline 与 candidate 在 `--selftest` 上差别巨大**（7/7 vs 1/7），建议在题面里把边界形状纳入正式门槛（当前仅作"迭代期自检"），否则"特化到目标形状换 3% 收益"这条捷径是免费的。
3. **K2 的 `--selftest` 在出厂 baseline 上就是失败的**（尾块 padding 语义，`Akk.copy_` 63 vs 64），建议修正出厂 baseline，避免候选人被既有缺陷干扰。

---

## 9. 未能复验的项

- K1 中间演进值 1.622 / 1.579 ms（未提交中间版本 kernel）。
- cube 版 cumsum 4.603 ms、cube time 309.6us、"vector cumsum ≈125us" 的隔离估算、指令级 profiling 表格（无对应代码可测，方法可信、数值未验证）。
- K2 报告中 9.466 ms 的 baseline 来源条件（无法还原其测量现场）。

---

## 10. 安全提醒（与本次评分无关，但同批交付物）

`kda_test/interview_en_answer/devcontainer_root_key` 是**与 `interview_en/` 中在用的私钥逐字节相同**的活密钥，且已随 commit `28a98dc935` **提交并推送到远端** `github.com:zhouhuaman/sglang`。后续 commit 删除文件并不能消除历史记录。建议：轮换该密钥对 → 用 `git filter-repo`/BFG 清除历史 → 强制推送。另外该目录与候选人可见的 `interview_en/` 并列，容易误发（内含完整答案与逐题优化报告）。

---

## 11. 复现命令

```bash
# 宿主侧（脚本未跟踪，勿进交付包）
docker exec triton-ascend-env-zhm bash /docker/zhm/0505_skill_test/sonnet/sglang/kda_test/_verify_answer_inner.sh 3        # 目标 case：正确性 + base/cand 成对性能
docker exec triton-ascend-env-zhm bash /docker/zhm/0505_skill_test/sonnet/sglang/kda_test/_verify_answer_extra_inner.sh 3  # 边界 selftest + 第二轮性能

# 单点复核（以 K2 为例，容器内）
mkdir -p /tmp/kda_verify && cd /tmp/kda_verify
source /docker/zhm/0505_skill_test/sonnet/sglang/kda_test/interview_en_answer/env.sh 3
msprof --output=/tmp/kda_verify/k2_base \
  --application="python3 /docker/zhm/0505_skill_test/sonnet/sglang/kda_test/interview/k2_token_parallel/test.py --perf --repeats 7 --warmup 3"
python3 /docker/zhm/0505_skill_test/sonnet/sglang/kda_test/interview/k2_token_parallel/test.py --report /tmp/kda_verify/k2_base
```

测量前务必 `npu-smi info` 选空闲卡（本次全程用 card 3；card 4~7 被占用）。
