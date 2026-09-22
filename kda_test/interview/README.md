# KDA 算子优化 · 面试题包

> 这是 **6 道开放题**（K1 门控累计 / K2 窗口打分 / K3 块求逆 / K4 重算 w·u /
> K5 状态递推 / K6 输出合成，对应 KDA 6-kernel 拆分的每一环）：每题给你一个**Triton kernel**（昇腾 910B2 上，
> 由你复现基线、profile、提出并落地优化，或给出可信的"已到当前约束极限"论证。
> 每题完整题面（算子介绍 / 数学 / 已知瓶颈 / 官方基线 / 一条 msprof 指令 / 契约）在
> **各自目录的 `PROBLEM.md`** 里，本文件只讲**环境**与**交付物**。

```
interview/
├── README.md                          ← 本文件（环境 + 交付物）
├── env.sh                             ← 容器环境准备（source 一次）
├── GRADER.md                          ← 【仅面试官】评分细则（勿发给候选人）
├── TEST_REPORT.md                     ← 2026-09-07 验收测试报告（环境/版本、精度、性能、AscendC 对比）
├── k1_gate_chunk_cumsum/
│   ├── PROBLEM.md                     ← 题面（K1）
│   ├── gate_chunk_cumsum_kernel.py    ← 你要读/改的 kernel（唯一改动点）
│   └── test.py                        ← 正确性门槛 + 性能复现
├── k2_token_parallel/                 ← 题面 K2（同上结构）
├── k3_inter_solve/                    ← 题面 K3
├── k4_recompute_w_u/                  ← 题面 K4
├── k5_delta_rule_h/                   ← 题面 K5
├── k6_gla_output/                     ← 题面 K6
└── bench/                             ← 【开发者用，勿发候选人】6-kernel 统一
                                         拉通 bench（全链正确性 + msprof 分段
                                         计时 + ASCENDC 融合算子对照），kernel
                                         从 ../k*_*/ 导入；用法见 bench/README.md
```

> 包内 kernel 已并入当前工具链的修复版（与 `kda_test/design/<op>/src` 同步）：
> K2 对角块收拢回退 hm2（满宽写 scratch + driver `torch.gather` —— `tl.gather`
> 在 triton-ascend 3.2.1 下对 dot 输出数值错误 max_diff≈0.32）；K5 外积与快照
> store 改为 CANN 9.1 可编译形态（输入侧转置 + 统一 2D store）。每题设计深度
> 见各自目录 `PROBLEM.md` 末尾「设计创新点与深度解析」。

**统一目标 case**（六题共用，评分口径）：`D_KV128_H96_T16384`
→ B=1, T=16384, H=96, K=V=128。所有性能以该 case 为准，具体基线与测量命令见各 `PROBLEM.md`。

---

## 1. 测试环境

### 1.1 连接（VS Code Remote-SSH，免密）

| 项目 | 值 |
|---|---|
| 服务器 | `116.204.40.238`（端口 22） |
| 用户 | `root` |
| 密钥 | `~/.ssh/devcontainer_root_key`（管理员发你，`chmod 600`，勿外传/勿入库） |
| 开发容器 | `triton-ascend-env-zhm`（进容器即完整 Triton/昇腾环境） |

首次把下面内容追加到本地 `~/.ssh/config`：

```
Host triton-env
    HostName 116.204.40.238
    User root
    Port 22
    IdentityFile ~/.ssh/devcontainer_root_key
    ConnectTimeout 30
```

VS Code 装 **Remote - SSH** 扩展 → `F1` → `Remote-SSH: Connect to Host...` → 选 `triton-env`
（首次选 Linux、确认 host key）。左下角变 `SSH: triton-env` 后，终端里进开发容器：

```bash
docker exec -it triton-ascend-env-zhm bash
```

容器内预置：Python 3.11.15、CANN 9.0.0、torch 2.7.1 + torch_npu + **triton 3.2.1 (Ascend)**、
8× 昇腾 910B2。`/docker`、`/data` 为宿主机与容器共享目录——代码在 VS Code 里编辑
（打开 `/docker/zhm/0505_skill_test/sonnet/sglang/kda_test/interview`），命令在容器终端跑即可。

### 1.2 用卡规矩

进容器先看空闲卡，**别与他人撞同一张卡**（并发会报 `ERR00100`/`Resource_Busy`）：

```bash
npu-smi info                     # 看各卡 AICore 占用率，挑空闲的一张
```

### 1.3 准备环境（每个新终端一次）

```bash
cd /docker/zhm/0505_skill_test/sonnet/sglang/kda_test/interview
source env.sh                    # 也可 source env.sh 4 —— 固定只用卡 4
```

`env.sh` 做三件事：source CANN `set_env.sh`、把 torch/torch_npu 动态库加入
`LD_LIBRARY_PATH`、设 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`（缺一 torch_npu 会加载失败）。
它会打印环境与卡状态，看到即可。验证：

```bash
python3 -c "import torch, torch_npu, triton; print(torch.__version__, triton.__version__)"
```

> 迭代提示：每道题只需改 `<题目录>/<op>_kernel.py`（唯一改动点，名字 = 目录下半部：
> `gate_chunk_cumsum_kernel.py` / `token_parallel_kernel.py` / `inter_solve_kernel.py` /
> `recompute_w_u_kernel.py` / `delta_rule_h_kernel.py` / `gla_output_kernel.py`），
> 不动 `test.py`/`env.sh`。改完用题面里的命令复测。

---

## 2. 交付物（markdown）

1. **复现基线**：目标 case 的 msprof 每调用均值（ms），附你跑的那条 msprof 命令 +
   `python3 test.py --report <dir>` 的输出（数字需落在题面给的官方区间内）。
2. **瓶颈论点**：你用什么证据（msprof op_summary / metric / 消融实验）支撑你的判断。
3. **改动**：你改了 kernel 哪里、为什么（一句话一个假设）。
4. **收益**：同一口径（同卡、同 repeats/warmup、同会话）改前 vs 改后 ms/调用 + 精度
   数字。**无收益或认为已收敛**：给出支撑"已到当前约束极限"的证据链——诚实的负结果 +
   收敛论证不劣于凑出来的小收益。
5. **放弃的方向**：你试过但放弃的优化及一句话原因（展示思考广度）。

**测量纪律**（对比是否可信的关键）：
- 性能一律以 msprof `Task Duration(us)` **每调用均值**为准；`test.py --perf` 自报的
  wall-clock 只作快照，不作评分。
- 你的收益对照**你自己复现的基线**，不跨机器/跨时段硬比；每次测前 `npu-smi info`
  确认空闲卡。
- 正确性以各题 `PROBLEM.md` 的 PASS 门槛为准（默认 `python3 test.py` 出 PASS）。

**不可破坏的契约**（各题 PROBLEM.md 详列，违规即无效解）：对外默认 `chunk_size=64`
（K2/K3 另 `sub_chunk_size=16`）不得改——各算子在上游 KDA 链路互为上下游，chunk 边界
共用同一份 64；fp32 主契约；精度保持 <1e-2 的量级。
