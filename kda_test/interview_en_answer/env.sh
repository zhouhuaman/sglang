#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# KDA interview-package environment setup: source once inside the triton-ascend-env-zhm
# container to run the three problems.
#
# Usage (under kda_test/interview_en):
#   source env.sh                # prepare CANN + torch/torch_npu dynamic libs + backend switch
#   source env.sh 4              # same, and pin the card via ASCEND_RT_VISIBLE_DEVICES=4
#
# Key points:
#   * Must source CANN set_env.sh, add torch/torch_npu's lib dirs to LD_LIBRARY_PATH, and set
#     TORCH_DEVICE_BACKEND_AUTOLOAD=0, otherwise torch_npu cannot load (torch's DEVICE_BACKEND
#     auto-load conflicts with torch_npu).
#   * Card selection: inside the container first run `npu-smi info` to find an idle card, then pass
#     an argument to pin one, to avoid colliding with a concurrent user (concurrency reports
#     ERR00100 / Resource_Busy). No argument = no pin (device 0 by default).
#   * This file changes no code and may be sourced repeatedly.

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$_HERE"

: "${SET_ENV:=/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [ -f "$SET_ENV" ]; then
  # shellcheck disable=SC1090
  source "$SET_ENV"
fi

# triton-ascend needs torch / torch_npu's dynamic libs on LD_LIBRARY_PATH
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

echo "[env] CANN + torch/torch_npu ready → run python3 <problem-dir>/test.py"
python3 -c "import torch, torch_npu, triton; print('  torch', torch.__version__, '| triton', triton.__version__)" 2>/dev/null || true
npu-smi info 2>/dev/null | sed -n '1,12p' || true
