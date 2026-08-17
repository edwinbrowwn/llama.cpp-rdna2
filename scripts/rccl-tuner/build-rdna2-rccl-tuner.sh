#!/usr/bin/env bash
set -euo pipefail
SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROCM_PATH=${ROCM_PATH:-/opt/rocm/core-7.14}
CXX=${CXX:-g++}
"$CXX" -std=c++17 -O2 -fPIC -shared -D__HIP_PLATFORM_AMD__ \
  -I"$ROCM_PATH/include" -L"$ROCM_PATH/lib" \
  -Wl,-rpath,"$ROCM_PATH/lib" \
  -o "$SELF_DIR/libnccl-tuner-rdna2-v620.so" \
  "$SELF_DIR/rccl-tuner-v620.cpp" -lamdhip64
readelf -Ws "$SELF_DIR/libnccl-tuner-rdna2-v620.so" | grep -q 'ncclTunerPlugin_v6'
printf 'RCCL_TUNER_BUILD=PASS\noutput=%s\n' "$SELF_DIR/libnccl-tuner-rdna2-v620.so"
