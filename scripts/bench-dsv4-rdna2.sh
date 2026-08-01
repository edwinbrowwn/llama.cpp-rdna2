#!/usr/bin/env bash
# Build (and optionally run) the synthetic DSV4 RDNA2 kernel micro-benchmark.
# Never touches the canonical build directory.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROCM_PATH="${ROCM_PATH:-/opt/rocm/core-7.14}"
CXX="$ROCM_PATH/llvm/bin/clang++"
OUT_DIR="${OUT_DIR:-/tmp/dsv4-bench}"
TARGET_ARCH="${TARGET_ARCH:-gfx1030}"

mkdir -p "$OUT_DIR"

export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"

"$CXX" -x hip --offload-arch="$TARGET_ARCH" -O3 -DNDEBUG -std=gnu++17 \
    -funsafe-math-optimizations \
    -DGGML_USE_HIP -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 \
    -I"$ROOT_DIR/ggml/src/ggml-cuda" -I"$ROOT_DIR/ggml/include" -I"$ROOT_DIR/ggml/src" \
    "$ROOT_DIR/ggml/src/ggml-cuda/bench/bench-dsv4-rdna2.cu" \
    -L"$ROCM_PATH/lib" -lamdhip64 -o "$OUT_DIR/bench-dsv4-rdna2"

echo "built: $OUT_DIR/bench-dsv4-rdna2"

if [[ "${RUN:-0}" == "1" ]]; then
    "$OUT_DIR/bench-dsv4-rdna2"
fi