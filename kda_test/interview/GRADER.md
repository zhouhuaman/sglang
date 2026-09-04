# GRADER —— 面试官评分细则与参考答案

> **勿发给候选人。** 本文件给三个 open-ended 题各自的真实瓶颈结论、已知死胡同与
> 打分要点。候选人可见的只有 README（环境 + 交付物）+ env.sh + 各自题目目录
> （题面在各目录 `PROBLEM.md`）。GRADER 勿随包外发。
>
> 证据细节可到 `../design/{inter_solve,token_parallel,delta_rule_h}/OPTIMIZATION_LOG.md`
> 与 `../design/unified/ANALYSIS.md` 翻阅（面试包若被单独拷出则这些不在候选环境里）。

---

## 0. 三题共通：可复现的前提

1. **官方基线自证**：候选应给出目标 case 一次 msprof `Task Duration(us)` 每调用均值，
   落在表中区间才算"复现成功"：
   K2 ≈9.5–9.8ms / K3 ≈14.4–14.9ms / K5 ≈9.1–9.5ms（单 kernel 隔离，±10%）。
2. **同口径比较**：收益必须对照**他自己复现的基线**（同卡、同 repeats/warmup、
   同会话）。跨时段/跨卡对比不可信。
3. **正确性门槛**：`python3 test.py`（默认 BT/BC=64/16）目标 case `max_diff<1e-2`。
4. **契约完整（最高优先级）**：交付 kernel 必须仍在默认 `chunk_size=64`
   （K2/K3 另 `sub_chunk_size=16`）下通过。**防坑**：个别候选人可能把 `chunk_size=128`
   写进默认值——那样 triton-vs-torch 在自洽的 128 下依然 PASS，但破坏了 KDA 六算子
   共享的 BT=64 chunk 契约（下游对不上），属于**无效解**。收包时在**干净副本**上跑
   原默认值 + `git diff` 核对默认参数未被改。

---

## 1. 通用打分框架（每题 100）

| 维度 | 权重 | 要点 |
|---|---|---|
| 复现与测量纪律 | 20 | msprof 口径正确、基线落区间、对比同口径；会用 `--report` 而非 wall-clock |
| 诊断质量 | 30 | 用证据（msprof metric / 消融 / 结构推理）定位，而非盲试参数 |
| 优化落地 | 35 | 改动有假设支撑；同一口径下收益可信；精度保 1e-2 |
| 收敛论证（若走此路） | 35 | 声称"已到极限"须有证据链而非感觉 |
| 表达 | 15 | 假设→实验→结论讲得清；能说出放弃方向及原因 |

> 注意：由于基线是**多轮收敛后的最终版**，多数参数级 lever 已耗尽。**一个"诚实的
> 负结果 + 有理有据的收敛论证"不劣于一个凑出来的小收益。** 强候选人反而要能判断
> 哪些方向不成立。

---

## 2. 每题真实答案速览（面试官参考）

### 题 K2 — token_parallel

**真实瓶颈**：内存延迟 / 低利用率（非带宽、非饱和）。msprof 各 pipe 全 <37%，
mac 仅 ~7%，aic_scalar ~33%，aiv 36%。CPU 侧每 head 的 2D 访存地址生成是剩余标量成本；
`tl.gather` 收拢对角块后已无参数级杠杆（HM/NW/NS 全扫 ≈9.76ms 平）。

**已是收敛配置**：HM=16（head-merge，摊每 CTA 标量 setup）、Route A 数学变换
`exp2(g[i]-g[j])→exp2(g[i])·exp2(-g[j])` 批量 tl.dot、chunked grid 消除 grid 超限
（上游 1 CTA/token/head 在 T=16384 会超 coreDim 65535）。

**开放方向（真杠杆）**：
- BT=128 chunk 统一（把 NT 减半、提高每 CTA 有效工作、摊薄启动/地址成本）——但要
  保 BT=64 契约（§0.4），故更可能是"内部把两个 64 合并处理、输出仍按 64 写"；
- 进一步削减 head 维度地址生成 / 提升访存并行度以喂饱延迟。

**防坑**：fp16 无收益（后端 upcast）；改布局转置类方案无效（非带宽受限）。

### 题 K3 — inter_solve

**真实瓶颈**：aic_scalar ~48%、aiv 低，aiv_cyc=1250M 主导——**不是标量饱和**，而是
**6 个串行 dot 的逆链**（Phase 3 的 `Ai_10→Ai_20/Ai_21→Ai_31→Ai_30` 依赖链）加上每
(64,head) CTA 的重标量 setup。HM/NW/NS/NP 全扫 ≈14.7ms 平（NP=3 截断逆已最优；
NP=2 的 18.2ms 是平台怪癖，dot 越多 pipeline 越好）。

**已是收敛配置**：单 kernel 融合（三阶段合一）、重复平方截断逆 npow=3、HM=16、nw=4、
`empty` 缓冲 + 无掩码 store（消除 2×402MB memset）、fp32。

**开放方向（真杠杆）**：
- **缩短 6-dot 串行依赖链**：Phase 3 合并逆存在代数重排/更高阶逆估计以并行化
  部分链式 matmul 的空间（关键路径是 A 的求逆，不是 Phase 1 的并行块）；
- BT=128 重构（同上，保契约）；每 CTA 标量 setup 摊薄（head-merge 进一步）。

**防坑**：改布局/带宽类无效；期望明显低于 14.4ms 的结构性改动需跨 chunk 契约变更，
若不保契约则无效。

### 题 K5 — delta_rule_h

**真实瓶颈**：**256-chunk 串行递推**每迭代延迟（aic_scalar ~59% 是表象；chunk 链无法
并行，cube 利用率 3–4%）。BV=32→64→128 扫描证明 BV=V 单调最优（每 chunk 2 dot）；
num_warps 1–16、num_stages 1–2 几乎无影响。Round3 把 4 dot/chunk→2 dot/chunk 是
唯一大收益（9.78→9.09ms）。`tl.range(num_stages=3)` 软流水再 −0.4ms（9.51→9.11）。

**已是收敛配置**：BV=V=128（整 V 驻留一个 CTA）、单 K-tile（K=128 整 K 一个 tile →
每 chunk 2 dot）、`tl.range(NT, num_stages=3)`、K=64 flat-store vs K≠64 2D-store 分路径。

**开放方向（真杠杆）**：
- 递推本质上串行（chunk c+1 需要 chunk c 的状态）——**block 级并行受制于序列依赖**；
  除非把 delta-rule 更新写成可块化形式（matrix-geometric / 半环上的前缀和类分解），
  否则 NT=256 链就是下界。这是本题最有价值也最难的方向；
- 每 chunk 迭代内的标量/延迟削减（load 提前、快照 store 与主链解耦）。

**防坑**：BT=128 被同事试过报 6.4ms，那是**破坏 chunk 契约的假象**（下游对不上），
在 BT=64 下不成立——可用来考察候选人是否会被"6.4ms"诱惑。

---

## 3. 参考评分锚点（每档口语化）

- **A（90+）**：干净复现；提出并落地了一个**有 msprof 证据支撑**的结构性优化，收益
  可信且精度达标；或给出了"受串行依赖/标量关键路径限制已收敛"的完整证据链。
  能讲清为什么某类优化（带宽/布局/fp16）无效。
- **B（75-90）**：复现 OK，诊断方向对，收益小/无但过程严谨；或优化有效但证据口径
  有瑕疵（用 wall-clock 报数、跨会话对比）。
- **C（60-75）**：复现勉强、主要在调参数碰运气，无结构判断；能保契约与精度。
- **D（<60）**：破坏契约（默认 BT/BC 被改、用自洽 BT=128 蒙混）、精度做坏、或
  wall-clock 冒充 msprof。

## 4. 收包 checklist

- [ ] 干净副本跑 `python3 test.py` → 目标 case PASS（默认 chunk_size=64）。
- [ ] `git diff`（或文件 diff）确认只动了 `*_kernel.py`，且默认 `_BT/_BC`/chunk_size 未改。
- [ ] 拿到一条自报 msprof 命令 + `--report` 输出，均值落官方区间 ±10%。
- [ ] 若有优化：同口径改前 vs 改后两组 `--report` 数字 + 各自 PASS。
- [ ] 精度未劣化到 ~1e-2 附近（看自报 max_diff）。
