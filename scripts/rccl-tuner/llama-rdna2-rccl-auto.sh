#!/usr/bin/env bash
# Development/production-launch wrapper for the external RDNA2 RCCL policy.
set -euo pipefail
SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${GGML_HIP_RCCL_WORKTREE:-$(cd "$SELF_DIR/../.." && pwd)}
SERVER_BIN=${LLAMA_SERVER_BIN:-$REPO_ROOT/build/bin/llama-server}
CALIBRATOR=${GGML_HIP_RCCL_CALIBRATOR:-$SELF_DIR/rdna2-rccl-calibrator.py}
PLUGIN=${GGML_HIP_RCCL_PLUGIN:-$SELF_DIR/libnccl-tuner-rdna2-v620.so}
CACHE=${GGML_HIP_RCCL_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/llama.cpp/rdna2-rccl-cache.json}
EVIDENCE=${GGML_HIP_RCCL_CALIBRATION_EVIDENCE:-${TMPDIR:-/tmp}/llama-rdna2-rccl-calibration-$$}
MODE=${GGML_HIP_RCCL_TUNE:-auto}
DRY_RUN=0
ARGS=()
MODEL=${GGML_HIP_RCCL_CALIBRATION_MODEL:-}

while (($#)); do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        -m|--model) MODEL=${2:?missing model after $1}; ARGS+=(-m "$2"); shift 2 ;;
        --model=*) MODEL=${1#*=}; ARGS+=("$1"); shift ;;
        --) shift; ARGS+=("$@"); break ;;
        *) ARGS+=("$1"); shift ;;
    esac
done

run_server() {
    if ((DRY_RUN)); then
        printf 'server=%q\nmode=%q\n' "$SERVER_BIN" "$MODE"
        printf 'args='; printf '%q ' "${ARGS[@]}"; printf '\n'
        printf 'NCCL_TUNER_PLUGIN=%q\n' "${NCCL_TUNER_PLUGIN-}"
        printf 'GGML_HIP_RCCL_TUNE=%q\n' "${GGML_HIP_RCCL_TUNE-}"
        return 0
    fi
    exec "$SERVER_BIN" "${ARGS[@]}"
}

if [[ "$MODE" != auto && "$MODE" != off && "$MODE" != force ]]; then
    echo "[rdna2-rccl] invalid GGML_HIP_RCCL_TUNE=$MODE; falling back to Auto" >&2
    MODE=off
fi

# Never override explicit user collective policy.
for var in NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS NCCL_NTHREADS; do
    if [[ -n "${!var-}" ]]; then
        echo "[rdna2-rccl] $var is user-set; skipping automatic RCCL policy" >&2
        MODE=off
        break
    fi
done

if [[ "$MODE" == off ]]; then
    unset NCCL_TUNER_PLUGIN GGML_HIP_RCCL_TUNE
    run_server
    exit $?
fi

if [[ ! -x "$SERVER_BIN" || ! -f "$CALIBRATOR" || ! -f "$PLUGIN" ]]; then
    echo "[rdna2-rccl] launcher artifact missing; falling back to RCCL Auto" >&2
    unset NCCL_TUNER_PLUGIN GGML_HIP_RCCL_TUNE
    run_server
    exit $?
fi

if [[ "$MODE" == force ]]; then
    export GGML_HIP_RCCL_TUNE=force
    export NCCL_TUNER_PLUGIN="$PLUGIN"
    run_server
    exit $?
fi

if [[ -z "$MODEL" || ! -f "$MODEL" ]]; then
    echo "[rdna2-rccl] no usable model path for auto calibration; falling back to RCCL Auto" >&2
    unset NCCL_TUNER_PLUGIN GGML_HIP_RCCL_TUNE
    run_server
    exit $?
fi
mkdir -p "$(dirname "$CACHE")" "$EVIDENCE"
python3 "$CALIBRATOR" --mode auto \
    --worktree "$REPO_ROOT" \
    --model "$MODEL" --binary "${LLAMA_BENCH_BIN:-$REPO_ROOT/build/bin/llama-bench}" \
    --plugin "$PLUGIN" --cache "$CACHE" --evidence "$EVIDENCE" \
    --tokens "${GGML_HIP_RCCL_CALIBRATION_TOKENS:-128}" \
    --reps "${GGML_HIP_RCCL_CALIBRATION_REPS:-3}" \
    >"$EVIDENCE/controller.json"
selected=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"])' "$EVIDENCE/controller.json")
if [[ "$selected" == Ring/LL/3-hot ]]; then
    export GGML_HIP_RCCL_TUNE=force
    export NCCL_TUNER_PLUGIN="$PLUGIN"
    echo "[rdna2-rccl] selected Ring/LL/3-hot; policy frozen before server graph initialization" >&2
else
    unset NCCL_TUNER_PLUGIN GGML_HIP_RCCL_TUNE
    echo "[rdna2-rccl] selected RCCL Auto" >&2
fi
run_server
