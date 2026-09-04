#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# KDA 面试包环境准备：在 triton-ascend-env-zhm 容器内 source 一次即可跑三个题。
#
# 用法（在 kda_test/interview 下）:
#   source env.sh                # 准备 CANN + torch/torch_npu 动态库 + 后端开关
#   source env.sh 4              # 同上，并把卡固定到 ASCEND_RT_VISIBLE_DEVICES=4
#
# 关键点:
#   * 必须先 source CANN set_env.sh、并把 torch/torch_npu 的 lib 加进
#     LD_LIBRARY_PATH、设 TORCH_DEVICE_BACKEND_AUTOLOAD=0，否则 torch_npu 无法加载
#     (torch 的 DEVICE_BACKEND auto-load 会与 torch_npu 冲突)。
#   * 卡的选择: 进容器先 `npu-smi info` 看空闲卡, 传参数固定到某张卡, 避免并发撞卡
#     (并发会报 ERR00100 / Resource_Busy)。不传参数=不限定(默认 device 0)。
#   * 本文件不改任何代码，可重复 source。

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$_HERE"

: "${SET_ENV:=/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [ -f "$SET_ENV" ]; then
  # shellcheck disable=SC1090
  source "$SET_ENV"
fi

# triton-ascend 需要 torch / torch_npu 的动态库在 LD_LIBRARY_PATH
TORCH_LIB=$(python3 -c 'import os,sysconfig; p=os.path.join(sysconfig.get_paths()["purelib"],"torch","lib"); print(p)' 2>/dev/null || true)
TORCHNPU_LIB=$(python3 -c 'import os,sysconfig; p=os.path.join(sysconfig.get_paths()["purelib"],"torch_npu","lib"); print(p)' 2>/dev/null || true)
for lib in "$TORCH_LIB" "$TORCHNPU_LIB"; do
  if [ -d "$lib" ]; then
    export LD_LIBRARY_PATH="$lib:$LD_LIBRARY_PATH"
  fi
done
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

if [ -n "$1" ]; then
  export ASCEND_RT_VISIBLE_DEVICES="$1"
  echo "[env] ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
fi

echo "[env] CANN + torch/torch_npu 就绪 → 可跑 python3 <题目录>/test.py"
python3 -c "import torch, torch_npu, triton; print('  torch', torch.__version__, '| triton', triton.__version__)" 2>/dev/null || true
npu-smi info 2>/dev/null | sed -n '1,12p' || true
