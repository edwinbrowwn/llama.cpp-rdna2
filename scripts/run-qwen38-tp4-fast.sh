#!/usr/bin/env bash
# Launch qualified four-GPU Qwen3.8 TP4 performance profiles.
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PROFILE=${1:-}
ACTION=${2:-}
if [[ -z "$PROFILE" || -z "$ACTION" ]]; then
    cat >&2 <<'EOF'
usage: run-qwen38-tp4-fast.sh <exact|max> <bench|server|exec> [arguments...]

  exact  Q8 target; byte-identical vocab-sharded output optimization
  max    auto20 Q4S8 target plus qualified gfx1030 native optimizations

Examples:
  ./scripts/run-qwen38-tp4-fast.sh exact bench -p 512 -n 128 -r 5
  ./scripts/run-qwen38-tp4-fast.sh max server -c 8192 --parallel 4
  ./scripts/run-qwen38-tp4-fast.sh exact exec ./build/bin/llama-cli -m "$Q8_MODEL" ...
EOF
    exit 2
fi
shift 2

Q8_MODEL=${Q8_MODEL:-/home/edwin/models/qwen38-27b-q4s8/unsloth-q8/Qwen3.8-27B-Q8_0.gguf}
Q4S8_MODEL=${Q4S8_MODEL:-/home/edwin/models/qwen38-27b-q4s8/qwen38-27b-mtp-q4s8-auto20-v1/model/Qwen3.8-27B-MTP-Q4S8-auto20.gguf}

export HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-10.3.0}
export HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-1}
export GGML_CUDA_DISABLE_GRAPHS=1
export GGML_CUDA_ALLREDUCE=nccl
export GGML_TP_SHARDED_OUTPUT=1
export GGML_TP_VOCAB_SHARDED_OUTPUT=1
export LD_LIBRARY_PATH="$ROOT/build/bin:/opt/rocm/core-7.14/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ulimit -s 8192

case "$PROFILE" in
    exact)
        MODEL=$Q8_MODEL
        unset GGML_HIP_GFX1030_NATIVE
        unset GGML_HIP_GFX1030_GDN_SIBLING_FUSION
        unset GGML_HIP_GFX1030_Q8_CACHE
        unset GGML_HIP_GFX1030_Q8_CACHE_TELEMETRY
        unset GGML_HIP_RDNA2_BF16_HIDDEN_ALLREDUCE
        ;;
    max)
        MODEL=$Q4S8_MODEL
        export GGML_HIP_GFX1030_NATIVE=1
        export GGML_HIP_GFX1030_GDN_SIBLING_FUSION=1
        export GGML_HIP_GFX1030_Q8_CACHE=1
        unset GGML_HIP_GFX1030_Q8_CACHE_TELEMETRY
        unset GGML_HIP_RDNA2_BF16_HIDDEN_ALLREDUCE
        ;;
    *)
        echo "unknown profile: $PROFILE (expected exact or max)" >&2
        exit 2
        ;;
esac

[[ -r "$MODEL" ]] || { echo "missing model: $MODEL" >&2; exit 1; }
[[ -x "$ROOT/build/bin/llama-bench" && -x "$ROOT/build/bin/llama-server" ]] || {
    echo "missing built llama-bench/llama-server under $ROOT/build/bin" >&2
    exit 1
}
export QWEN38_MODEL=$MODEL

echo "QWEN38_TP4_PROFILE=$PROFILE" >&2
echo "QWEN38_MODEL=$MODEL" >&2

case "$ACTION" in
    bench)
        exec "$ROOT/build/bin/llama-bench" \
            -m "$MODEL" -b 2048 -ub 512 -ngl 999 -sm tensor \
            -dev ROCm0/ROCm1/ROCm2/ROCm3 -ts 1/1/1/1 -fa on "$@"
        ;;
    server)
        exec "$ROOT/build/bin/llama-server" \
            -m "$MODEL" -ngl all --split-mode tensor --fit off \
            -dev ROCm0,ROCm1,ROCm2,ROCm3 -ts 1,1,1,1 -fa on "$@"
        ;;
    exec)
        [[ $# -gt 0 ]] || { echo "exec requires a command" >&2; exit 2; }
        exec "$@"
        ;;
    *)
        echo "unknown action: $ACTION (expected bench, server, or exec)" >&2
        exit 2
        ;;
esac
