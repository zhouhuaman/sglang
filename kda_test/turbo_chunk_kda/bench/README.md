# turbo_chunk_kda/bench —— KDA 6-kernel 统一拉通 bench + ASCENDC 融合算子对照

> **本目录从 `../turbo_gate_chunk_cumsum/` ~ `../turbo_gla_output/` 的六算子 kernel 导入**
> （与各算子目录 `*_kernel.py` 同源，turbo_chunk_kda 目录即单算子/拉通 bench 的唯一工作区），
> 完成 6 个算子**串成全链**的正确性（模式 A）与 msprof 分段计时（模式 B）。
> 与 `kda_test/design/unified` 的关系：同一套框架；本目录把 kernel 导入路径换成
> turbo 各算子目录，并适配 turbo 命名（K1 `turbo_gate_chunk_cumsum_triton`、
> K6 `turbo_gla_output_triton`）。

## 数据流（全链口径）

```
g       = K1(x, A_log, dt_bias)                       # 门控激活 + chunk 前缀和
Aqk_d, Akk  = K2(q, k, g, beta, scale)                # 块内（对角线）得分
Aqk_nd, Akk_inv = K3(q, k, g, beta, Akkd=Akk, scale)  # 块间得分 + 下三角逆
w, u, kg = K4(k, v, beta, A=Akk_inv, gk=g)            # 解耦表示
h, v_new = K5(kg, w, u, gk=g, initial_state=init, idx)# 跨 chunk 状态递推
o        = K6(q, v_new, g, Aqk=Aqk_d+Aqk_nd, h=h)     # 输出合成
```

每个 case 先用 torch 元算子链算出 K1..K6 各自输入，再让 K_torch 与 K_triton 在
**完全相同输入**下对比 max_diff（`<1e-2` PASS）。

## 用法（容器内；先 `source ../env.sh`）

```bash
cd kda_test/turbo_chunk_kda/bench
python3 bench.py --start 105 --limit 1          # ① 目标 case 全链正确性（模式 A）
python3 bench.py --limit 3                      # 冒烟：前 3 个 case

bash run_cpu.sh --msprof ./prof_target --start 105 --limit 1 \
     --repeats 5 --warmup 2                     # ② msprof 采集（性能唯一口径）
python3 per_case_profile.py --latest-dir ./prof_target --mean   # → results.csv
python3 analyze_results.py --pivot-case D_KV128_H96_T16384      # 加速比矩阵

python3 time_kernels.py                         # ③ 迭代用 wall-clock 计时 + 精度
python3 time_kernels.py --kernel K3             #    只测单 kernel（改完快速验证）
bash run_all_msprof_local.sh                    # ④ 全量 106 case msprof（分批）
```

- 用例表 `cases_meta.json`（106 case，含目标 case `D_KV128_H96_T16384`）；
  `gen_cases.py` 可重建；张量由 `bench.py` 固定种子即时生成，不入库。
- 全链（K1→K6）任何 kernel 改动后先 `python3 test.py`（单算子，各题目录）PASS，
  再跑本目录模式 A 复核全链精度，最后 msprof 报数 —— 与 TEST_REPORT.md 口径一致。

## ASCENDC 融合算子对照（官方基线来源）

- `prof_chunk_kda_fwd_fused.py`：vllm-ascend `ChunkKdaFwd`（Ascend C 融合算子，
  fp16/bf16）的 msprof 计时脚本 —— 六分拆 triton 版的对照基线即由此测量。
  需 `vllm-ascend-community` 环境（脚本内 `_REPO` 硬编码本机路径，迁移时改它）。
- `chunk_kda_fwd_fused_bench.md`：融合算子测法/结果文档。

## 环境与注意

- 本目录脚本假定 triton-ascend-env-zhm 容器（CANN set_env + torch/torch_npu 动态库 +
  `TORCH_DEVICE_BACKEND_AUTOLOAD=0`），`run_cpu.sh`/`time_run.sh` 自带环境设置并默认
  屏蔽物理卡 0；`run_all_msprof.sh` 为宿主机 docker exec 驱动版。
- msprof 产物（`prof*`、`results*.csv` 等）不入库，见本目录 `.gitignore`。
