#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# 在 triton-ascend-env-zhm 容器内跑 time_kernels.py（wall-clock 迭代优化用）。
# 环境设置与 run_cpu.sh 完全一致（CANN set_env + LD_LIBRARY_PATH + 屏蔽物理卡 0）。
#
# 用法:
#   bash time_run.sh                              # 全 6 kernel, 目标 case
#   bash time_run.sh --kernel K3                  # 只测单 kernel（改完快速验证）
#   bash time_run.sh --repeats 10 --warmup 3
#
# 宿主侧: docker exec -w <unified目录> triton-ascend-env-zhm bash time_run.sh ...

set -e

: "${SET_ENV:=/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [ -f "$SET_ENV" ]; then
  # shellcheck disable=SC1090
  source "$SET_ENV"
fi

TORCH_LIB=$(python3 -c 'import os,sysconfig; p=os.path.join(sysconfig.get_paths()["purelib"],"torch","lib"); print(p)' 2>/dev/null || true)
TORCHNPU_LIB=$(python3 -c 'import os,sysconfig; p=os.path.join(sysconfig.get_paths()["purelib"],"torch_npu","lib"); print(p)' 2>/dev/null || true)
for lib in "$TORCH_LIB" "$TORCHNPU_LIB"; do
  if [ -d "$lib" ]; then
    export LD_LIBRARY_PATH="$lib:$LD_LIBRARY_PATH"
  fi
done
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

# 屏蔽物理卡 0（曾硬故障 open 507033）；用 VISIBLE_DEVICES 覆盖。
: "${VISIBLE_DEVICES:=1,2,3,4,5,6,7}"
export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"

: "${DEVICE:=}"
DEV_ARGS=()
if [ -n "$DEVICE" ]; then DEV_ARGS+=(--device "$DEVICE"); fi

cd "$(dirname "$0")"

python3 time_kernels.py "$@" "${DEV_ARGS[@]}"
