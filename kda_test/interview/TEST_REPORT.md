# KDA 六算子 Triton 独立实现 — 验收测试报告

> 测试日期：2026-09-07 ｜ 测试目录：`kda_test/interview/`（面试题包 + 单算子/拉通工作区）
> 被测对象：K1..K6 六个算子的独立实现（`interview/k*_*/*_kernel.py`，纯 torch + triton，
> 不依赖 sglang / vllm-ascend 任何代码，与 `kda_test/design/<op>/src/` 同源）。
> 性能唯一口径：msprof `Task Duration(us)` 每调用均值（设备侧 kernel 时间）。

---

## 1. 测试环境与版本

### 1.1 硬件与系统

| 项 | 值 |
|---|---|
| 主机 | `A5-29`（Linux 5.4.0-125-generic x86_64） |
| NPU | 昇腾 **Ascend 950PR**（A5，`ascend950pr_9579`）× 6 可见，测试用卡 3 |
| 驱动 | 25.7.rc1.b999（ascendhal 7.35.23） |

### 1.2 软件版本（关键）

| 组件 | 版本 | 说明 |
|---|---|---|
| CANN | **9.1.0** | `ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0` |
| Python | 3.11.10 | `/usr/local/python3.11.10` |
| torch | 2.10.0+cpu | CPU 版构建 + torch_npu 后端（非 GPU 版） |
| torch_npu | 2.10.0.post4 | |
| triton-ascend | **3.2.1** | 华为 Ascend fork；`triton.__version__` 显示 3.2.0 为 base 包版本号，实际安装 `triton_ascend-3.2.1.dist-info` |
| msprof | CANN 9.1.0 自带 | `op_summary_*.csv` 的 `Task Duration(us)` 为权威耗时 |

### 1.3 工具链变更记录（重要背景）

| 时间 | 事件 | 影响 |
|---|---|---|
| 2026-08-31 | a5 收敛态测量（design 报告，36.64ms） | 本报告数字与之对照的基准 |
| 2026-09-02 | 本机 triton-ascend 被升级到 3.2.2 | **K2 hm2 编译失败**（`ub overflow`，需 1901568 > 可用 1769472 bits）；**K3 NP=3 编译失败**（`cbuf overflow`，需 5242880 > 可用 4194304 bits） |
| 2026-09-07 | 全局回退 **triton-ascend 3.2.1**（华为源 wheel；3.2.2 wheel 备份 `/root/toolchain_backup/`） | K2/K3 恢复可编译，6/6 PASS |
| 2026-09-07 | **K6 修复**：因果掩码 `tl.where(fp32 比较, ...)` 改原生 bool 比较 | 3.2.1 对「fp32 掩码 → tl.dot 输入」误编译（块内路径错误 max_diff≈0.34）；修复后 3.2.1/3.2.2 双版本均 PASS（1.2e-7），逐位等价 |

---

## 2. 测试对象与统一口径

六算子数据流（KDA chunked attention 拆解，BT=64 chunk、BC=16 sub-chunk）：

```
g       = K1(x, A_log, dt_bias)                        # 门控激活 + chunk 前缀和 → [B,T,H,K]
Aqk_d, Akk  = K2(q, k, g, beta, scale)                 # 块内（对角线 16×16）得分
Aqk_nd, Akk_inv = K3(q, k, g, beta, Akkd=Akk, scale)   # 块间得分 + 下三角逆
w, u, kg = K4(k, v, beta, A=Akk_inv, gk=g)             # 解耦表示
h, v_new = K5(kg, w, u, gk=g, initial_state, idx)      # 跨 chunk 状态递推（唯一串行）
o        = K6(q, v_new, g, Aqk=Aqk_d+Aqk_nd, h=h)      # 输出合成
```

**目标 case（评分口径，六题共用）**：`D_KV128_H96_T16384` — `B=1, T=16384, H=96, K=V=128`，
输入输出 **fp32**，NT=256，契约 **chunk_size=64 / sub_chunk_size=16 不可改**。

**正确性口径**：每题 `python3 test.py` = triton kernel vs 同文件 torch 元算子参考
（K2/K6 另有纯 torch CPU 逐 token 参考 `*_ref` 兜底），逐元素 `max_diff < 1e-2` 判 PASS。

**性能口径**：每题一条 msprof 指令（见下），解析 op_summary 按 triton kernel 名过滤取
`Task Duration(us)` 每调用均值（calls = warmup 3 + repeats 7 = 10 次）：

```bash
source ../env.sh                       # CANN 环境（可用 `env.sh 3` 固定卡 3）
python3 test.py                        # ① 正确性：PASS + max_diff
msprof --output=./prof_kX --application="python3 test.py --perf --repeats 7 --warmup 3" \
    && python3 test.py --report ./prof_kX     # ② 基线（Task Duration 每调用均值）
```

> 注：测试期间卡 3 存在 ~60% 的并发负载（共享机）；为排除扰动，本报告数字与
> 2026-08-31 独立会话的 design 报告对照，差异均 <1.5%，可复现性良好。

---

## 3. 精度测试结果（目标 case，2026-09-07）

| Kernel | 参考对比 | 实测 max_diff | 门槛 | 结果 |
|---|---|---|---|---|
| K1 gate_chunk_cumsum | `gate_chunk_cumsum_torch` | 1.37e-04 | <1e-2 | ✅ PASS |
| K2 token_parallel（hm2） | `token_parallel_torch`（Aqk/Akk） | 2.98e-07 | <1e-2 | ✅ PASS |
| K3 inter_solve（NP=3） | `inter_solve_torch`（Aqk/Akk_inv） | 4.77e-07 | <1e-2 | ✅ PASS |
| K4 recompute_w_u | `recompute_w_u_torch`（w/u/kg） | 2.98e-04 | <1e-2 | ✅ PASS |
| K5 delta_rule_h | `delta_rule_h_torch`（h/v_new/终态） | 8.94e-07 | <1e-2 | ✅ PASS |
| K6 gla_output（修复后） | `gla_output_torch`（o） | 1.19e-07 | <1e-2 | ✅ PASS |

K6 边界 shape 附加回归（`python3 test.py --selftest`，覆盖尾 chunk 不满 / K=V=32/128 /
V≠K / 多 head / 多 chunk / 多 batch）：

| case | shape (B,T,H,K,V) | max_diff | 结果 |
|---|---|---|---|
| tiny_k64 | 1,64,1,64,64 | 6.7e-08 | ✅ |
| tail_t63 | 1,63,2,64,64 | 6.0e-08 | ✅ |
| k128_tail | 1,127,2,128,128 | 6.0e-08 | ✅ |
| k32_long | 1,193,1,32,32 | 8.9e-08 | ✅ |
| h8_k64 | 1,256,8,64,64 | 1.0e-07 | ✅ |
| kv_mix | 1,128,2,64,128 | 6.0e-08 | ✅ |
| b2_k64 | 2,128,2,64,64 | 7.5e-08 | ✅ |

全部 6 kernel 目标 case + 7 边界 case **PASS**，且远优于 1e-2 门槛（最高 3e-4）。

---

## 4. 性能测试结果（目标 case，msprof 每调用均值）

| Kernel | ms/调用 | 说明 |
|---|---|---|
| K1 gate_chunk_cumsum | **1.373** | 纯访存 + 前缀和，无矩阵乘 |
| K2 token_parallel | **5.548** | hm2：整 chunk 大 dot + driver gather 收拢对角块 |
| K3 inter_solve | **10.696** | NP=3 截断逆，HM=16 |
| K4 recompute_w_u | **3.623** | 单 tile 直通 |
| K5 delta_rule_h | **11.431** | BV=V、NS=3 流水线、外积输入侧转置形态 |
| K6 gla_output | **3.912** | HM=16 双路单累加器（bool 掩码修复后） |
| **合计（6-kernel 链）** | **36.58 ms** | |

与 2026-08-31 design 报告（同机同工具链口径）对照：

| Kernel | 本报告 | 8/31 报告 | 差异 |
|---|---|---|---|
| K1 | 1.373 ms | 1.380 ms | +0.5% |
| K2 | 5.548 ms | 5.621 ms | +1.3% |
| K3 | 10.696 ms | 10.693 ms | −0.03% |
| K4 | 3.623 ms | 3.609 ms | −0.4% |
| K5 | 11.431 ms | 11.425 ms | −0.05% |
| K6 | 3.912 ms | 3.914 ms | +0.05% |
| 合计 | 36.58 ms | 36.64 ms | +0.15% |

> 差异全部 <1.5%：确认环境还原（triton-ascend 3.2.1）与测量的可复现性。
> 参考：同 case 下 torch_npu 元算子链（6 算子拼接）约 **~1.9 s**（8/31 报告，
> K5 torch 单算子约 697 ms），triton 6-kernel 相对 torch 拼接加速 **~50×**。

---

## 5. 与 Ascend C 手写融合版本的对比

> 对照对象：vllm-ascend 仓库手写 **ASCENDC 融合算子 `ChunkKdaFwd`**（Gate/Prepare/
> PostWu/FwdH/Finalize 融合为一次 aclnn 调用）。测量文档见
> `interview/bench/chunk_kda_fwd_fused_bench.md`（2026-08-31，同机同 case），
> 方法学提炼见 `kda_test/design/KDA_LEARNING_GUIDE.md` §7。

### 5.1 口径差异（先对齐再比）

| 维度 | Triton 6-kernel 分拆（本报告） | ASCENDC 融合算子 |
|---|---|---|
| 物理形态 | 6 个独立 triton kernel，逐调用 msprof 隔离求和 | 1 次 aclnn 调用 = 1 个 L2 Transpose + **4 个物理 kernel**（#0 32.9ms + #1 2.4 + #2 5.4 + #3 3.0），按调用求和 |
| 精度位宽 | **fp32** 主契约（门槛 1e-2，实测 1e-7~3e-4） | fp16 / bf16 |
| 中间张量 | Aqk/Akk/Akk_inv/w/u/kg/h/v_new 落 HBM（每 kernel 独立读写） | kernel 内寄存器/UB 传递，免中间量全局往返 |
| 正确性基准 | torch 元算子 / CPU 逐 token 参考，目标 case 全 T=16384 比对 | CPU 参考抽查 T=1024（max_abs=1.9e-6，通过） |

### 5.2 总耗时对比（目标 case D_KV128_H96_T16384）

| 实现 | 位宽 | 每调用耗时 | 相对 |
|---|---|---|---|
| **Triton 6-kernel 分拆** | fp32 | **36.58 ms**（本报告，msprof） | **1.00×** |
| ASCENDC 融合（bf16） | bf16 | 37.22 ms（event，≈msopprof） | 1.02× |
| ASCENDC 融合（fp16） | fp16 | **43.67 ms**（msopprof） | 1.19× |

### 5.3 结论（结合 KDA_LEARNING_GUIDE §7）

1. **同数量级，triton 分拆版略快**：triton 6-kernel 用 **fp32**（精度更高）做到
   36.58ms，比 ASCENDC 融合算子 fp16（43.67ms）快 ~19%，与 bf16 版（37.22ms）持平。
   融合算子虽然省去了中间张量的 HBM 往返，但其 4-launch 结构中主 kernel #0 占 75%
   且融合带来的寄存器/控制复杂度没有被 6 分拆版各自独立标量优化（head-merge、
   loop-invariant 提升、无掩码 store 等）抵消。
2. **两者均 memory/流水线受限而非 Cube 受限**：ASCENDC 版有效算力粗估仅 ~7 TFLOPS
   （A5 fp16 理论值远高于此），dtype 行为（bf16 比 fp16 快 15%）呈带宽敏感特征；
   triton 版各 kernel cube 利用率普遍 <20%（见 KDA_LEARNING_GUIDE §3/§8），主瓶颈是
   标量管线与访存 —— 两侧瓶颈同源，性能差距主要来自实现形态而非算法。
3. **ASCENDC 融合版的优势场景**：bf16 部署（Kimi K3 实际路径）37.2ms 与 triton fp32
   持平；若 triton 侧也压到 bf16/fp16，融合版融合中间量的价值才会显现 —— 目前 triton
   版把「中间量落 HBM」的成本用每 kernel 更优的 tile/掩码设计补偿掉了。
4. **工程/可验证性差异（选型参考）**：
   - triton 分拆版：每算子可独立 profile/调优/验证（本题面 6 题的由来），中间量可
     复算核对，精度链路可分段定位；代价是多 kernel launch 与中间张量内存带宽。
   - ASCENDC 融合版：单算子部署简单、无中间张量，但单 kernel 调试面大（4 launch
     内部不可见），精度问题难分段定位，且 A5 无 runtime custom-op 能力需绕过门控加载。

### 5.4 遗留与建议

- ASCENDC 版全 T=16384 的精度未抽查（仅 T=1024），两侧严格同 case 精度对比待补；
- 全链 106 case 矩阵（`interview/bench/bench.py`）与拉通 wall-clock
  （`interview/bench/time_kernels.py`）可在本环境直接复跑，本报告聚焦目标 case 口径；
- triton-ascend 3.2.2 下 K2/K3 的 UB/cbuf 收紧问题已记录（§1.3），3.2.2 wheel 备份于
  `/root/toolchain_backup/`，venv 验证环境 `/tmp/ta321`（3.2.1）/`/tmp/ta322`（3.2.2）。

---

## 6. 结论

interview 包 6 个 kernel 在还原后的 a5 工具链（CANN 9.1.0 + triton-ascend 3.2.1）上
**正确性 6/6 PASS**（含 7 个边界 shape），目标 case 总耗时 **36.58 ms**（fp32），与
8/31 独立测量一致（<1.5%）；横向对比手写 ASCENDC 融合算子（fp16 43.67ms / bf16
37.22ms）**同数量级且 fp32 口径下更优**。K6 的 3.2.1 编译器误编译问题已定位并修复
（bool 掩码，双版本验证无回归），修复与基线校准已同步到 interview 与 design 两份。
