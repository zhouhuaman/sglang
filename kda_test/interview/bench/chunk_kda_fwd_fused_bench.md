# vllm-ascend 融合 ChunkKdaFwd 算子 — 性能测试与结果分析

> 测试对象：`/data/autotriton/sekd/vllm-ascend-community/vllm-ascend/csrc/attention/chunk_kda_fwd`
> 配套脚本：`prof_chunk_kda_fwd_fused.py`（本目录）
> 测试日期：2026-08-31

## 1. 背景与目标

vllm-ascend 仓库中的 `chunk_kda_fwd` 是 KDA（Key Delta Attention）chunkwise forward 的融合
ASCENDC 算子，对齐不涉及 CP 切分的 FLA `chunk_kda_fwd` 顶层语义（Gate、Prepare、PostWu、FwdH、
Finalize 在一个 `ChunkKdaFwd` 中完成，见算子 README）。

目标回归 case（与 `design/unified/cases_meta.json` 中定义一致）：

```
"D_KV128_H96_T16384": {
  "B": 1, "T": 16384, "H": 96, "K": 128, "V": 128,
  "group": "D",
  "desc": "K=V=128 (目标 case): H=96,T=16384"
}
```

即：BSND 布局，B=1，T=16384，H=HV=96（GQA group=1），K=V=128，scale=K^-0.5。
算子设计文档（`docs/design.md`）明确「性能结论只使用 msopprof」。

## 2. 测试环境

| 项 | 值 |
| --- | --- |
| NPU | Ascend950PR（A5, ascend950pr_9579）×6 可见 |
| CANN | 9.1.0（`ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0`） |
| torch / torch_npu | 2.10.0+cpu / 2.10.0.post2 |
| vllm_ascend | 本仓库 editable 构建 `0.19.1rc2.dev1789+gadfc2a2f5` |
| 计时工具 | torch.npu Event（快速对照）+ msprof / msopprof（权威） |

## 3. 构建与算子加载

### 3.1 构建

算子此前未编译（`torch.ops._C_ascend.chunk_kda_fwd` 未注册），本机从零构建：

```bash
# 1) catlass submodule（A5 构建依赖）
git submodule update --init --recursive csrc/third_party/catlass

# 2) 标准构建（SOC 自动识别 ascend950pr_9579 → build_aclnn.sh 编 26 个 A5 自定义算子 + torch 扩展）
pip install -e . --no-deps --no-build-isolation
```

产物：

- `vllm_ascend/vllm_ascend_C.cpython-311-x86_64-linux-gnu.so`（torch 绑定）
- `vllm_ascend/_cann_ops_custom/vendors/custom_transformer/`（含 `chunk_kda_fwd` 的
  op_proto / op_api / op_impl(ai_core/tbe/ascend950 kernel)）

### 3.2 运行时加载（A5 特有）

A5 的硬件 profile **不含** `RUNTIME_CUSTOM_OPS` 能力，`vllm_ascend.utils.enable_custom_op()`
会直接返回 False 而不注册算子。测试脚本绕过该门控：

```python
from vllm_ascend.utils import bootstrap_custom_op_env
bootstrap_custom_op_env(include_vendor_lib=True)   # 设置 ASCEND_CUSTOM_OPP_PATH + LD_LIBRARY_PATH
import vllm_ascend.vllm_ascend_C                   # 注册 torch.ops._C_ascend.chunk_kda_fwd
```

注意：不要 import `vllm_ascend.meta_registration`（本构建会因 `bgmv_expand` 不存在而报错，
与本次测试无关）。

### 3.3 环境踩坑记录

1. **PEP517 隔离构建卡死**：`python3 setup.py develop` 走 `pip install -e . --use-pep517`，
   需下载 ~2GB 构建依赖（torch/transformers/triton-ascend 等），本机网络对 >50MB 文件
   不稳定（连接反复中断、pip 挂起无进展）。解决：`--no-deps --no-build-isolation`，
   全局环境已有全部所需构建依赖（triton-ascend 3.2.1 即可，3.2.2 仅为运行时版本 pin）。
2. **editable 安装冲突**：机器上存在另一用户（t00869793）的旧 editable 安装，其 finder
   在 site-packages 中排在前面，从仓库外运行时 `import vllm_ascend` 会解析到别人 checkout
   （无 chunk_kda_fwd）。解决：脚本开头把本仓库路径 `sys.path.insert(0, ...)`（普通
   sys.path 优先于 editable finder）。
3. **A5 internal format 警告**：`allow_internal_format=True` 在 A5 无效（仅 A2/A3 支持），
   不影响结果。

## 4. 测试方法

脚本 `prof_chunk_kda_fwd_fused.py` 提供三种模式：

| 模式 | 命令 | 用途 |
| --- | --- | --- |
| event 计时 | `python3 prof_chunk_kda_fwd_fused.py [--dtype] [--chunk-size] [--iters]` | 快速对照（含 aclnn 调度开销） |
| msopprof 采集 | `msprof --application="python3 prof_chunk_kda_fwd_fused.py --iters 200" --output=/tmp/kda_prof` | 权威 kernel 耗时 |
| msprof 解析 | `python3 prof_chunk_kda_fwd_fused.py --parse /tmp/kda_prof` | 解析 op_summary |
| 正确性抽查 | `python3 prof_chunk_kda_fwd_fused.py --check` | 小 T 下对照 CPU 参考实现 |

输入构造：q/k/v 为 `randn×0.05`，g 为 `-rand×0.05`（raw gate，`use_gate_in_kernel=False`），
beta 为 `sigmoid(randn)`，均与 nightly 测试同分布。

msprof 解析的关键：**A5 上单次 `aclnnChunkKdaFwd` 调用实际发射 1 个 layout Transpose
kernel + 4 个 `KdaChunkForward_ChunkKdaFwd` kernel**（见 §5.2）。解析按 Transpose 行分界
把同一调用的多个 kernel 归组求和，得到 per-call 总耗时；同名 kernel 再按名称聚合
给出阶段分解。

## 5. 测试结果

### 5.1 目标 case 汇总（B=1, T=16384, H=96, K=128, V=128, BSND）

| 配置 | msopprof per-call | event avg | 吞吐 |
| --- | --- | --- | --- |
| **fp16, chunk=64（目标）** | **43.67 ms** | 43.72 ms | ~375 K tokens/s |
| bf16, chunk=64 | — | 37.22 ms | ~440 K tokens/s |
| fp16, chunk=128 | — | 51.24 ms | ~320 K tokens/s |

正确性抽查（T=1024 同 H/K/V/chunk 路径，CPU 参考实现对照）：max_abs=1.9e-6，通过。

### 5.2 目标 case kernel 分解（msopprof，211 次调用统计，耗时极稳定）

| kernel | 次数/调用 | avg | min | max |
| --- | --- | --- | --- | --- |
| `KdaChunkForward_ChunkKdaFwd` #0（主 kernel） | 1 | 32.88 ms | 32.86 ms | 33.83 ms |
| `KdaChunkForward_ChunkKdaFwd` #1 | 1 | 2.36 ms | 2.35 ms | 2.48 ms |
| `KdaChunkForward_ChunkKdaFwd` #2 | 1 | 5.39 ms | 5.38 ms | 5.40 ms |
| `KdaChunkForward_ChunkKdaFwd` #3 | 1 | 3.03 ms | 3.02 ms | 3.06 ms |
| `TransposeAiCore_Transpose`（L2 布局转换） | 1 | 0.017 ms | 0.016 ms | 0.019 ms |
| **per-call 合计** | 5 | **43.67 ms** | 43.64 ms | 44.74 ms |

注：单次调用内 4 个 kernel 的位置模式完全稳定（#0≈33ms 恒为第一个），说明这是 L2 按
阶段/分块固定发射的 4 次物理 launch，不是负载均衡的切分。

## 6. 结果分析

### 6.1 物理 kernel 拆分与 README 表述的差异

算子 README 描述「Gate、Prepare、PostWu、FwdH 和 Finalize 均在一个物理 `ChunkKdaFwd`
L0 内完成」，但 A5 实测单次调用发射 **4 个同名 `KdaChunkForward` kernel + 1 个
`TransposeAiCore` kernel**（BSND→BNSD 布局转换由 L2 以独立 aclnn Transpose 完成，
对应 README 中「BSND/TND 由 L2 使用 `l0op::Transpose` 转为内部 BNSD/NTD」）。

对性能结论的影响：per-call 总耗时需按 4 个 kernel 求和（43.67ms），若只看单个 kernel
会得到 2.4~32.9ms 的错误结论；4 次 launch + 1 次 Transpose 的调度开销合计约 100µs 量级
（各 kernel 的 wait 时间未计入），相对 43.7ms 可忽略，瓶颈在 kernel 执行本身。

### 6.2 主 kernel 占比 75%

主 kernel（#0，32.9ms）占单次调用 75% 的耗时。若后续优化，优先分析 #0 的 Cube 利用率
与 chunk 间状态串行依赖（256 chunks × 96 heads 的并行度充足，理论上不是并行度受限，
更可能是访存/流水线停顿）。

### 6.3 dtype 对比：bf16 比 fp16 快 15%

| dtype | 耗时 | 相对 fp16 |
| --- | --- | --- |
| fp16 | 43.7 ms | 1.00× |
| bf16 | 37.2 ms | 0.85× |

符合带宽敏感特征：q/k/v 总量 ~1.2GB，bf16 与 fp16 位宽相同但内部 FP32 计算路径更
省（fp16 需额外转 FP32 处理），或 gate/累加路径差异所致。对 Kimi K3 实际部署（bf16）
有利。

### 6.4 chunk_size 对比：64 优于 128

| chunk | 耗时 | 相对 chunk=64 |
| --- | --- | --- |
| 64 | 43.7 ms | 1.00× |
| 128 | 51.2 ms | 1.17× |

chunk 内 Aqk/Akk/inv_akk 是 O(chunk²)（inv_akk 是 O(chunk³)），chunk 翻倍使 intra-chunk
成本约翻倍，而 inter-chunk 状态更新的节省不足以抵消。chunk=64 是当前目标场景的正确选择。

### 6.5 有效算力粗估（供参考，非精确）

按每 chunk-head 的 matmul 量粗估（C=64, K=V=128；不含 gate elementwise/exp2）：

```
per chunk-head ≈ 4·2·C²·K (Aqk/Akk/w/u) + 3·2·C·K·V (v_new/h/o_inter) + o_local 2·C²·V + inv_akk ~0.5M
               ≈ 12 MFLOPs
总 FLOPs ≈ 12M × (T/C) × H = 12M × 256 × 96 ≈ 0.3 TFLOPs
有效算力 ≈ 0.3e12 / 0.04367s ≈ 7 TFLOPS
```

A5 fp16 理论算力远高于此，说明 kernel 远未饱和（访存带宽、FP32 内部计算、gate 等
elementwise 未计入，实际数值会更高但数量级不变）。结论：**该 case 当前是
memory/流水线受限而非 Cube 计算受限**，与 6.3 中 dtype 带宽敏感的表现一致。

### 6.6 部署视角

- 吞吐 ~375K tokens/s（fp16）/ 440K tokens/s（bf16），即单 token-head 约 28ns；
- 对 128K 长序列 prefill（Kimi K3 场景），单层单 head 组合耗时需按层数叠加，该算子
  本身不是 O(T²)，chunk 化后随 T 线性扩展，可放心用于长序列。

## 7. 复测步骤

```bash
cd /data/autotriton/sekd/sglang/kda_test

# 1) 正确性抽查（可选）
python3 prof_chunk_kda_fwd_fused.py --check

# 2) event 计时（快速）
python3 prof_chunk_kda_fwd_fused.py                                  # fp16, chunk=64
python3 prof_chunk_kda_fwd_fused.py --dtype bf16                     # bf16, chunk=64
python3 prof_chunk_kda_fwd_fused.py --chunk-size 128                 # fp16, chunk=128

# 3) msopprof 权威采集 + 解析
rm -rf /tmp/kda_prof
msprof --application="python3 prof_chunk_kda_fwd_fused.py --iters 200 --warmup 10" \
       --output=/tmp/kda_prof
python3 prof_chunk_kda_fwd_fused.py --parse /tmp/kda_prof
```

## 8. 已知限制

- event 计时包含 aclnn 调度/launch 开销，与 msopprof 的差异在 ~0.1% 以内（本 case
  launch 开销相对 kernel 可忽略），结论以 msopprof 为准。
- 本次仅测 `use_gate_in_kernel=False`（raw gate）路径；`safe_gate`/`use_gate_in_kernel=True`
  /`initial_state`/变长序列未覆盖。
- FLOPs 为手算粗估，未含 gate 计算；如需精确值建议结合 op_summary 的 AIC 指标。
