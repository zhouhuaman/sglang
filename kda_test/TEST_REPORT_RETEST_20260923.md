# KDA turbo 三形态 · 目标 case 性能复测与全量回归报告

- **日期**：2026-09-23
- **目标 case**：`D_KV128_H96_T16384`（B=1, T=16384, H=96, K=V=128，chunk=64）
- **对照基线**：`turbo_chunk_kda{,_fp16,_bf16}/TEST_REPORT.md`（2026-09-11 交付版）
- **被测对象**：当前工作区三棵树（含"参考优化迁移"改动，尚未提交）

---

## 0. 结论速览

| 指标 | 旧报告（fp32） | 本次复测（fp32） | 变化 |
|---|---|---|---|
| 目标 case 链式合计 | 36642.05 us | **34408.88 us** | **−6.1%** |
| 全链加速比（vs torch 链） | 24.86× | **26.44×** | +1.6× |
| 单算子隔离合计 | 36.587 ms | **34.346 ms** | **−6.1%** |

- **fp32 / fp16 / bf16 三棵树全部拿到收益**：隔离合计分别 −6.1% / −3.6% / −6.6%。
- 收益来自两处改动：**K1 一维 grid（−16%）**、**K3 `NP=2`（−18.6%）**；其余算子逐位持平（K4/K5 零变化）。
- fp16 树 K2 **变慢 26.5%**（4.122→5.213 ms），这是修复其**精度缺陷**的必然代价（旧版该行精度不达标），非性能回归。
- 全量 106 case（636 行）回归：**fp32 / bf16 均 627/636**，失败清单为既有的声明式限制（4 行 grid 超限 + 5 行 K=32 不支持），**无新增数值回归**。
- **定位到交付版既有 507015 缺陷的一个未记录触发场景**：fp16 树 K4 在 `T % 64 ≠ 0` 时触发 AI Core trap（详见 §6。非新缺陷 —— 交付文档已记录同族缺陷，但只覆盖满块情形）。**目标 case（T%64=0）不受影响**，fp32/bf16 树亦不受影响；已给出并验证补丁，fp16 回归由 431/636 恢复到 **627/636**。

> **口径警告**：本报告全部数据在 **conda `autotriton` 环境（triton-ascend 3.2.1）** 下测得，与交付 TEST_REPORT 口径一致。
> 默认 `/usr/local/python3.11.10/bin/python3` 是 **triton-ascend 3.2.2+dev**，同一 kernel 会给出差异很大的数字（见 §5.3），**两套工具链的数字不可混用**。

---

## 1. 测试环境

| 项 | 值 |
|---|---|
| 硬件 | Ascend950PR ×6（本机仅 0–5；卡 0 Alarm / 卡 2 Critical，本次测量全部使用健康卡） |
| CANN | 9.1.0 |
| Python | `/data/anaconda3/envs/autotriton/bin/python3`（交付报告指定环境） |
| triton-ascend | **3.2.1**（`triton.__version__` 报 3.2.0） |
| torch / torch_npu | 2.10.0+cpu / 2.10.0.post1.dev20260709 |
| vector_core_num | 56 |

**测量口径**（与旧报告一致）：

- 性能数字取自 **msprof `op_summary` 的 `Task Duration(us)`**。
- **链式**（§3）：`bash run_cpu.sh --msprof <dir> --start 105 --limit 1 --repeats 5 --warmup 2`，再用 `per_case_profile.py --latest-dir <dir> --mean` 取每算子均值。
- **单算子隔离**（§4）：`msprof --export=on --output=<dir> --application="python3 test.py --perf --repeats 7 --warmup 3"`，再用 `test.py --report <dir>` 汇总。
- 正确性：`bench.py`（模式 A），106 case × 6 kernel = 636 行。

> ⚠️ msprof 要求 `--output` 目录**预先存在**（需 `mkdir -p`），否则静默不产出 `op_summary_*.csv`。

---

## 2. 全量 106 case 回归（模式 A）

| 树 | 通过 | 失败 | 失败构成 |
|---|---|---|---|
| turbo_chunk_kda（fp32） | **627/636** | 9 | A_B4_H8_T131072 ×4、D_KV32_H4_T4096 ×5 |
| turbo_chunk_kda_bf16 | **627/636** | 9 | 同上，逐行一致 |
| turbo_chunk_kda_fp16（打补丁前） | 431/636 | 205 | **被 §6 的设备 trap 级联污染**，非数值问题 |
| turbo_chunk_kda_fp16（**补丁后，即当前树**） | **627/636** | 9 | **与 fp32/bf16 逐行一致** |

失败清单（三树完全相同，均为**已声明的结构性限制**）：

```
FAIL A_B4_H8_T131072 K2/K3/K4/K6   grid(flattened)=65536 超过 NPU coreDim 上限 65535
FAIL D_KV32_H4_T4096 K1/K2/K3/K4/K6 triton 不支持 K=32（BK=32 时 tl.dot 不稳定）
```

**无新增数值回归**：三棵树在 104/106 个 case 上全算子通过，剩余 2 个 case 是已知限制。

---

## 3. 目标 case 链式性能（msprof 模式 B，单位 us）

| 算子 | 旧报告(fp32) | **新 fp32** | 变化 | **新 fp16** | **新 bf16** |
|---|---|---|---|---|---|
| K1 gate_chunk_cumsum | 1375.96 | **1151.34** | −16.3% | 940.96 | 1152.93 |
| K2 token_parallel | 5599.91 | **5572.20** | −0.5% | 5216.83 | 3829.01 |
| K3 inter_solve | 10691.62 | **8706.22** | **−18.6%** | 8802.15 | 8799.57 |
| K4 recompute_w_u | 3632.41 | **3637.83** | +0.1% | 1999.33 | 3633.30 |
| K5 delta_rule_h | 11429.63 | **11427.24** | −0.0% | 11426.45 | 11426.24 |
| K6 gla_output | 3912.51 | **3914.06** | +0.0% | 2314.89 | 2396.15 |
| **合计** | **36642.05** | **34408.88** | **−6.1%** | **30700.62** | **31237.19** |
| torch 全链 | 910883.73 | 909637.99 | — | 911298.85 | 912021.98 |
| **加速比** | 24.86× | **26.44×** | +1.6× | **29.68×** | **29.20×** |

精度（链式，门槛 <1e-2）：fp32 最差 1.3e-04（K4）；fp16 最差 3.98e-03（K1）；bf16 最差 6.03e-03（K2）。**全部达标**。

> 旧报告只有 fp32 有链式表，fp16/bf16 仅记录隔离数据，故链式只能与 fp32 逐项对比。

---

## 4. 单算子隔离性能（单位 ms，独立进程 + msprof）

| 算子 | 旧 fp32 | 新 fp32 | 旧 fp16 | 新 fp16 | 旧 bf16 | 新 bf16 |
|---|---|---|---|---|---|---|
| K1 | 1.374 | **1.153** (−16.1%) | 1.195 | **0.943** (−21.1%) | 1.374 | **1.149** (−16.4%) |
| K2 | 5.567 | **5.539** (−0.5%) | 4.122 | 5.213 (**+26.5%**) | 3.826 | **3.830** (+0.1%) |
| K3 | 10.693 | **8.708** (−18.6%) | 10.790 | **8.802** (−18.4%) | 10.792 | **8.799** (−18.5%) |
| K4 | 3.614 | **3.613** (−0.0%) | 1.998 | **1.998** (0%) | 3.614 | **3.614** (0%) |
| K5 | 11.428 | **11.426** (−0.0%) | 11.428 | **11.428** (0%) | 11.428 | **11.429** (0%) |
| K6 | 3.911 | **3.907** (−0.1%) | 2.303 | **2.305** (+0.1%) | 2.381 | **2.384** (+0.1%) |
| **合计** | **36.587** | **34.346 (−6.1%)** | **31.836** | **30.689 (−3.6%)** | **33.415** | **31.205 (−6.6%)** |

隔离数据与链式数据逐项吻合（例如 fp32 K5：11.426 vs 11.427），说明两套测量互相印证。

---

## 5. 收益归因

### 5.1 K1 一维 grid：−16%（三树一致）

`K1_MODE=1`（一维展平 grid）替代交付版的三维 grid，消除核间调度开销：

| 树 | 交付版（三维 grid） | 新（一维 grid） | 变化 |
|---|---|---|---|
| fp32 | 1.374 ms | 1.148 ms | −16.4% |
| fp16 | 1.195 ms | 0.943 ms | −21.1% |
| bf16 | 1.374 ms | 1.149 ms | −16.4% |

副产品：一维 grid 顺带**解除 coreDim 65535 限制**（最大 case `A_B4_H8_T131072` 的 K1 由"预检拒绝"变为实测通过）。

### 5.2 K3 `NP=2`：−18.6%（三树一致）

`K3_NP` 控制截断级数：`NP=2` 比交付默认 `NP=3` 少 2 个 `tl.dot`。在 3.2.1 上的配置扫描（目标 case，msprof）：

| 配置 | 精度 max_diff | 门槛 | 耗时 |
|---|---|---|---|
| HM=16 / NP=3（交付默认） | 4.768e-07 | PASS | 10.692 ms |
| **HM=16 / NP=2（本次采用）** | **3.311e-04** | **PASS** | **8.711 ms (−18.5%)** |
| HM=32 / NP=3 | — | PASS | 10.800 ms |
| HM=8 / NP=3 | — | PASS | 10.850 ms |
| HM=1 / NP=2 | — | PASS | 11.209 ms |
| HM=1 / NP=3 | — | PASS | 13.156 ms |
| HM=16 / NP=1 | 4.463e-02 | **FAIL** | 6.822 ms（不可用） |

> **注**：早期在 triton-ascend **3.2.2+dev** 上曾观察到 HM>1 编译失败，据此把 `K3_HM` 改成了 1。
> 在交付口径 **3.2.1** 上复测证明该结论不成立：**HM=16 编译正常**，且 HM=16/NP=2 是最优组合（8.711 ms）。
> 已把 `_HM` 恢复为交付默认 **16**，仅采纳 `NP=2`。

### 5.3 工具链敏感度（不可忽视）

同一份 kernel 在 3.2.1 与 3.2.2+dev 上的耗时**方向都不一致**：

| 算子 | 3.2.1（交付口径） | 3.2.2+dev | 备注 |
|---|---|---|---|
| K1（一维 grid） | 1.148 ms | 1.479 ms | 3.2.2+dev 上更慢 |
| K2 | 5.567 ms | 9.323 ms | 3.2.2+dev 上慢 67% |
| K3（HM=1/NP=2） | 11.209 ms | 6.039 ms | 3.2.2+dev 上更快 |
| K5 | 11.428 ms | 7.142 ms | 3.2.2+dev 上更快 |

**结论**：任何性能结论都必须绑定工具链版本；本报告数字仅适用于 **triton-ascend 3.2.1**。

### 5.4 fp16 K2 变慢是"修复代价"，不是回归

fp16 树 K2 交付版 4.122 ms 是在**精度不达标**的前提下测得的；改用 fp32 dot 操作数后精度打标，代价是 **4.122 → 5.213 ms（+26.5%）**。
该 kernel 的其它 dtype 与 K4/K5/K6 全部零变化，确认无性能回归。

---

## 6. fp16 树 K4 在 `T % 64 ≠ 0` 时触发设备级 trap

> **与交付文档的关系（先读）**：507015 **不是新缺陷**。`turbo_chunk_kda_fp16/EXPERIMENT_FP16.md`
> （§K3）与 `turbo_chunk_kda_bf16/EXPERIMENT_BF16.md`（§2）已把它记录为 **triton-ascend 3.2.1
> 对"低精度 cube 操作数 × 深依赖 dot 链"的代码生成缺陷**，K3 正是因此采用"B 形态"
> （输入/输出低精度 + kernel 内 fp32 cube）规避。
>
> 但那两份文档同时下了"**浅链（NP=1）与独立 dot（K2/K4/K6）均正常**"的结论。本次复测表明
> 该结论**只对满块（`T % 64 == 0`）成立**：K4 的 dot 按上述分类属"独立 dot"，却在
> **非满块路径（`T%64≠0`）**上同样触发 507015。故本节是对既有记载的**触发场景补充**
> ——交付时未覆盖 `T%64≠0` 的 fp16 K4 ——**而不是发现了一个新缺陷**。

### 6.1 现象

fp16 树全量回归在 `B_T1023`（T=1023，`T%64=63`）的 **K4** 上崩溃，随后同进程的后续 case 全部 `SETUP_ERR`，表现为 205 行失败（级联污染）：

```
FAIL B_T1023 K4 RUN_ERR: npuSynchronizeDevice ... device error type 3, error code is 507015
FAIL B_T1025 K1 SETUP_ERR: copy_between_host_and_device_opapi ... 507015
```

开 `ASCEND_LAUNCH_BLOCKING=1` 后拿到根因：

```
aivec error exception, core id is 63, error code = 0
mte error info: 0x8e105a240002b367 ... errorStr: timeout or trap error. subErrType: 0x4
An error occurred in the kernel task, retCode=0x26, [aicore exception]
```

即 **AI Core vector 单元 trap**，不是普通的精度问题。

### 6.2 触发条件（已精确界定）

| case | T%64 | fp16 树 K4 | fp32 / bf16 树 K4 |
|---|---|---|---|
| B_T1536 | 0 | ✅ 通过 | ✅ |
| **D_KV128_H96_T16384（目标 case）** | **0** | **✅ 通过** | ✅ |
| B_T100000 | 32 | ❌ trap | ✅ |
| B_T1025 | 1 | ❌ trap | ✅ |
| B_T1023 | 63 | ❌ trap | ✅ |

**触发条件 = fp16 树（GM 侧 fp16）× K4 × `T % 64 ≠ 0`（即 `T_FULL=False` 分支）。**

### 6.3 定位过程与结论

| 实验 | 结果 | 结论 |
|---|---|---|
| 交付版 HEAD 树（`git archive a6e70bc56f`）同 case | **同样 trap** | **交付版既有缺陷，非本次改动引入** |
| 把 dot 操作数全改回 fp32（BISECT-A） | 仍 trap | 与 `tl.dot` 的 dtype 无关 |
| 输入张量 clamp 到 ±100（实测输入无 inf/nan，gk∈[−24.4, −0.03]） | 仍 trap | **与数值溢出无关** |
| bf16 树的 kernel 源码 + fp16 数据（只换 GM dtype） | 仍 trap | **触发点在 GM dtype=fp16 本身**（编译器/指令层） |
| `T%64≠0` 时输入转 fp32 计算 | **不再 trap** | workaround 有效 |

### 6.4 影响与 workaround

- **影响面**：仅 fp16 树、仅 K4、仅 `T%64≠0` 的 case。**目标 case `D_KV128_H96_T16384`（T%64=0）及 fp32/bf16 全树不受影响**。
- **workaround（已验证）**：在 K4 wrapper 里按 `T % chunk_size` 选择计算精度 ——

  ```python
  _dt = torch.float16 if (T % chunk_size == 0) else torch.float32
  k = k.to(_dt).to("npu");  v = v.to(_dt).to("npu")
  beta = beta.to(_dt).to("npu");  A = A.to(_dt).to("npu")
  ```

  配套使用自适应 dtype 的 kernel 写法（`b_vb = (b_v * b_b[:, None]).to(b_v.dtype)`、`b_kb.to(b_k.dtype)`）。
  实测 `B_T1023` / `B_T1025` 均不再 trap，且全量回归达到 **627/636**（与 fp32/bf16 失败清单完全一致）。
- **代价**：`T%64≠0` 的 case 在 fp16 树上退化为 fp32 计算（更慢，但正确性优先）。目标 case 为 `T%64=0`，**无此代价**。

---

## 7. 手动测试命令（三版本）

三个版本**目录名不同、命令完全相同**：

| 版本 | 目录（相对 `kda_test/`） | 全量回归应得 |
|---|---|---|
| fp32（基线） | `turbo_chunk_kda` | PASS **627/636** |
| fp16 | `turbo_chunk_kda_fp16` | PASS **627/636** |
| bf16 | `turbo_chunk_kda_bf16` | PASS **627/636** |

失败清单三树一致（`A_B4_H8_T131072` 的 K2/K3/K4/K6 共 4 行 grid 超限 + `D_KV32_H4_T4096` 的 K1/K2/K3/K4/K6 共 5 行 K=32 不支持）。

### 7.0 准备（每个 shell 一次）

```bash
export PATH=/data/anaconda3/envs/autotriton/bin:$PATH   # 必须：交付口径 triton-ascend 3.2.1
ROOT=/data/autotriton/sekd/sglang/kda_test
CARD=5                                   # 健康卡；先 npu-smi info（本机卡 0 Alarm、卡 2 Critical、卡 1 高温）
TREE=turbo_chunk_kda_fp16                # ← 三选一：turbo_chunk_kda / turbo_chunk_kda_fp16 / turbo_chunk_kda_bf16
source $ROOT/$TREE/env.sh $CARD          # 备好 CANN + torch/torch_npu + 固定卡
```

> 下面 §7.1–7.4 每段都从绝对路径起手，可**单独复制执行**，不分先后；§7.5 是这四段的自动化封装。

### 7.1 全链正确性（模式 A，6 个算子一次跑完）

```bash
cd $ROOT/$TREE/bench
python3 bench.py --start 105 --limit 1   # (a) 目标 case 全链正确性（106 条中末位，约 1 min）
python3 bench.py                         # (b) 全量 106 case（约 5 min，结果写 bench/correctness.csv）
```

> **(a) 是"一键测所有 kernel 正确性"的最小闭环**：目标 case 按 K1→K2→K3→K4→K5→K6 依次跑完
> 6 个算子，逐算子打印 `max_diff` 与 `OK/FAIL`，末尾 `PASS: 6/6 rows OK`。
> 索引 105 = `D_KV128_H96_T16384`。**(b)** 在此基础上铺满 106 条 case × 6 算子 = 636 格，末尾
> `PASS: 627/636`。两者都是**全链**（前一个算子的输出喂给下一个）；**单算子**版（不进链、每算子
> 独立进程）见 §7.3。

### 7.2 目标 case 链式性能（模式 B，msprof 口径，约 5 min）

```bash
cd $ROOT/$TREE/bench
VISIBLE_DEVICES=$CARD bash run_cpu.sh --msprof /tmp/prof --start 105 --limit 1 --repeats 5 --warmup 2
python3 per_case_profile.py --latest-dir /tmp/prof --mean     # → bench/results.csv
```

等价的不经 `run_cpu.sh` 的裸写法（需已 `source env.sh $CARD` 且**自己建目录**）：

```bash
cd $ROOT/$TREE/bench && rm -rf /tmp/prof && mkdir -p /tmp/prof
msprof --export=on --output=/tmp/prof \
    --application="python3 bench.py --msprof --start 105 --limit 1 --repeats 5 --warmup 2"
python3 per_case_profile.py --latest-dir /tmp/prof --mean
```

> 索引 105 = 目标 case `D_KV128_H96_T16384`（见 `cases_meta.json`）。
> `run_cpu.sh` 自带 `mkdir -p` 与 CANN 环境；但**若直接调 `msprof`（见 §7.4）则必须自己先建目录**。
> `run_cpu.sh` 的 `VISIBLE_DEVICES` 默认值是 `1,2,3,4,5,6,7`（屏蔽坏卡 0），**会覆盖** `env.sh` 设的卡 ——
> 所以这里要显式传 `VISIBLE_DEVICES=$CARD`。

### 7.3 单算子正确性（每棵树 × 6 算子）

```bash
K=turbo_gate_chunk_cumsum    # ← 六选一：turbo_gate_chunk_cumsum / turbo_token_parallel / turbo_inter_solve /
                             #           turbo_recompute_w_u / turbo_delta_rule_h / turbo_gla_output
cd $ROOT/$TREE/$K
python3 test.py --target     # 只测目标 case D_KV128_H96_T16384
python3 test.py              # 不传参 = 全 shape 扫描（含慢的 CPU 参考，耗时长）
```

### 7.4 单算子隔离性能（msprof）

```bash
cd $ROOT/$TREE/$K && rm -rf /tmp/pd && mkdir -p /tmp/pd   # msprof 要求目录预先存在，否则静默不产出
msprof --export=on --output=/tmp/pd --application="python3 test.py --perf --repeats 7 --warmup 3"
python3 test.py --report /tmp/pd          # 打印 calls/sum/mean（msprof Task Duration 均值）
```

### 7.5 一键脚本 `run_all_tests.sh`（§7.1–7.4 的封装）

```bash
cd /data/autotriton/sekd/sglang/kda_test

bash run_all_tests.sh --target                  # 三棵树 × 目标 case：① 全链 + ③ 6 个单算子，约 9 min
bash run_all_tests.sh                           # ① 换成全量 106 case（默认），约 21 min
bash run_all_tests.sh --case 105 --perf         # 目标 case + ② 全链 msprof + ④ 单算子隔离 msprof，约 36 min
bash run_all_tests.sh --dtype fp16 --case 105   # 只测 fp16 一棵树（单棵 ≈ 上列 ÷ 3）
bash run_all_tests.sh --case D_KV128_H96_T16384 # 按 case 名选（可前缀匹配，需唯一）
bash run_all_tests.sh --case 7,23,105           # 任意 case 集；脚本自动合并连续段
CARD=3 bash run_all_tests.sh --dtype bf16       # 换卡
```

| 参数 | 取值 | 默认 |
|---|---|---|
| `--dtype` | `fp32` / `fp16` / `bf16` / `all` | `all`（三棵树全跑） |
| `--case` | `all` / 索引 `105` / 名字 `D_KV128_H96_T16384` / 逗号列表 `7,23,105` | `all`（106 条全跑） |
| `--target` | 等价于 `--case 105` | — |
| `--perf` | 附带 ② 全链 msprof 与 ④ 单算子隔离 msprof | 关（只测正确性） |

四段输出分别标 `[①]` 全链正确性 / `[②]` 全链性能 / `[③]` 单算子正确性 / `[④]` 单算子隔离性能。
注意 ③/④ 的 `test.py` 只支持目标 case（`--target`）或全 shape 扫描（无参、很慢），**不受 `--case` 影响**。
`--case` 非 `all` 时 `bench.py` 走 `--start/--limit` 区间，故脚本会把索引集切成连续段逐段跑。
环境变量：`CARD`（默认 5）。默认 `all` + `all` 时末尾提示预期 `PASS 627/636`。

本次实测（卡 5，本机）：
- `--dtype fp32 --target` 单棵 **2 min 55 s**（① 40 s + ③ 6 算子约 2 min）→ 三棵约 9 min
- `--dtype fp16 --case 105 --perf` 单棵 **12 min**（② msprof 链约 4 min + ③④ 6 算子约 8 min）→ 三棵约 36 min
- ④ 实测值（fp16 目标 case，隔离 msprof，与 §4 逐项吻合）：K1 0.941 / K2 5.213 / K3 8.801 /
  K4 1.996 / K5 11.429 / K6 2.304 ms，合计 30.68 ms

---

## 8. 结论与建议

1. **目标 case 性能全面提升**：fp32 链式 36642 → **34409 us（−6.1%，26.44×）**，隔离 36.587 → **34.346 ms（−6.1%）**。
   另两个形态同步受益：fp16 隔离 −3.6%、bf16 隔离 −6.6%。
2. **收益来源明确且可解释**：K1 一维 grid（−16%）、K3 `NP=2`（−18.6%），其余算子逐位持平 —— 符合"最小改动"预期。
3. **K3 的交付配置问题已在交付口径下澄清**：3.2.1 上 HM=16 可编译，恢复交付默认 HM=16 并叠加 `NP=2`，是当前最优解。
4. **fp16 的 `T%64≠0` trap 是交付版既有的 507015 缺陷**（交付文档已记录该缺陷族，但只覆盖满块情形）。
   已按 §6.4 打补丁并随本次改动一并提交：fp16 全量回归 **431/636 → 627/636**，与 fp32/bf16 逐行一致；
   目标 case（`T%64=0`）走 fp16 原路径，性能零影响（1.991 → 1.998 ms，噪声内）。建议在交付文档中补记该触发场景。
5. **冗余代码已清理**（清单见 `REFERENCE_OPT_MIGRATION.md` §7.4）：三棵树删除从未被启动的死 kernel
   `_token_parallel_kernel_hm3`（各 80–86 行）、fp16 固化 fp32 dot 并拆除 `K2_DOT_F32` 开关。
   清理后 fp16 全量回归**复跑 627/636**，失败清单与 fp32/bf16 逐行一致。
6. **待办**：本次改动尚未提交；`REFERENCE_OPT_MIGRATION.md` 中基于 3.2.2+dev 的旧结论（K3 "出厂默认不可编译"、3.2.2+dev 链式表）已按本报告更正。
