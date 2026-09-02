#!/usr/bin/env bash
set -Eeuo pipefail
LABEL=${LABEL:?}
MODE=${MODE:?fixed|batch|auto}
PORT=${PORT:?}
RUN_ID=${RUN_ID:-iter4-active-batch}
PARALLEL=${PARALLEL:-2}
N_PREDICT=${N_PREDICT:-256}
SPEC_DEPTH=${SPEC_DEPTH:-4}
GRAPH_MODE=${GRAPH_MODE:-smooth}
STACK=${STACK:-mtp}
PROMPT_KIND=${PROMPT_KIND:-ordinary}
[[ $PORT =~ ^[0-9]+$ && $PORT -ne 8080 && $PORT -gt 1024 && $PORT -lt 65535 ]]
[[ $MODE == fixed || $MODE == batch || $MODE == auto ]]
[[ $GRAPH_MODE == smooth || $GRAPH_MODE == on ]]
[[ $STACK == mtp || $STACK == stacked ]]
[[ $PROMPT_KIND == ordinary || $PROMPT_KIND == repeat ]]
[[ $PARALLEL =~ ^[1-9][0-9]*$ && $N_PREDICT =~ ^[1-9][0-9]*$ ]]
[[ $SPEC_DEPTH =~ ^[1-4]$ ]]
WORK=${WORK:-/home/edwin/.ralph/mtp-dynamic-controller-work}
BUILD=${BUILD:-/home/edwin/.ralph/mtp-dynamic-controller-build-gfx1030}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/home/edwin/.ralph/mtp-dynamic-controller-artifacts}
OUT=$ARTIFACT_ROOT/$RUN_ID/$LABEL
MODEL=/home/edwin/models/qwen38-27b-magicquant-iq4-xs/Qwen3.8-27B-Quark-MXFP4-MQ-IQ4_XS_1-Generic.gguf
MMPROJ=/home/edwin/models/qwen38-27b-q4s8/autoround-q4-fast/mmproj-model.gguf
mkdir -p "$OUT"
PID=
cleanup() {
    rc=$?; trap - EXIT INT TERM; set +e
    if [[ -n ${PID:-} ]] && kill -0 "$PID" 2>/dev/null; then
        kill -TERM "$PID" 2>/dev/null
        for _ in $(seq 1 80); do kill -0 "$PID" 2>/dev/null || break; sleep .25; done
        kill -KILL "$PID" 2>/dev/null; wait "$PID" 2>/dev/null
    fi
    if fuser /dev/kfd >/dev/null 2>&1; then
        echo post_idle=no >&2; fuser -v /dev/kfd >&2; [[ $rc -ne 0 ]] || rc=76
    else
        echo post_idle=yes
    fi
    exit "$rc"
}
trap cleanup EXIT INT TERM
if fuser /dev/kfd >/dev/null 2>&1; then echo pre_idle=no >&2; fuser -v /dev/kfd >&2; exit 75; fi
if ss -ltn "sport = :$PORT" | grep -q LISTEN; then echo port_busy >&2; exit 74; fi
SPEC_ARGS=(--spec-type draft-mtp)
if [[ $STACK == stacked ]]; then
    SPEC_ARGS=(--spec-type draft-mtp,ngram-map-k4v --spec-ngram-map-k4v-size-n 12 --spec-ngram-map-k4v-size-m 48)
fi

DYN=()
if [[ $MODE == fixed ]]; then
    DYN=(LLAMA_SPEC_MTP_DYNAMIC_DEPTH=off)
elif [[ $MODE == batch ]]; then
    DYN=(LLAMA_SPEC_MTP_DYNAMIC_DEPTH=batch)
fi
export PATH=/opt/rocm/core-7.14/bin:/opt/rocm/core-7.14/llvm/bin:$PATH
export LD_LIBRARY_PATH=$BUILD/bin:/opt/rocm/core-7.14/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export SPEC_SIDECAR=1 HSA_NO_SCRATCH_RECLAIM=1 GGML_HIP_RDNA2_AUTO=1 GGML_HIP_SAFE_STATE_IO=1
export GGML_TP_SHARDED_OUTPUT=1 HSA_OVERRIDE_GFX_VERSION=10.3.0 HIP_VISIBLE_DEVICES=0,1,2,3
export LLAMA_SPEC_CYCLE_TRACE=1
if (( ${#DYN[@]} )); then
    export "${DYN[@]}"
else
    unset LLAMA_SPEC_MTP_DYNAMIC_DEPTH
fi
unset GGML_CUDA_DISABLE_GRAPHS GGML_HIP_GFX1030_SPEC_GRAPHS
if [[ $GRAPH_MODE == on ]]; then
    export GGML_HIP_GFX1030_SPEC_GRAPHS=1
elif [[ $SPEC_DEPTH != 4 ]]; then
    # The automatic smooth-profile selector is intentionally exact-width-four.
    # Fixed-depth schedule probes must use the same graph-off control.
    export GGML_CUDA_DISABLE_GRAPHS=1
fi
"$BUILD/bin/llama-server" \
  -m "$MODEL" -ngl all --split-mode tensor --tensor-split 1,1,1,1 \
  --device ROCm0,ROCm1,ROCm2,ROCm3 --flash-attn on --ctx-size 262144 \
  --batch-size 8192 --ubatch-size 4096 --host 127.0.0.1 --port "$PORT" \
  --temp 1 --top-p .95 --top-k 20 --min-p 0 --presence-penalty 0 --repeat-penalty 1 \
  --metrics --reasoning-effort xhigh --reasoning-preserve "${SPEC_ARGS[@]}" \
  --spec-draft-n-max "$SPEC_DEPTH" --spec-draft-p-min 0 --parallel "$PARALLEL" --spec-draft-ubatch-size 4096 \
  --cache-ram 65535 --mmproj "$MMPROJ" --ctx-checkpoints 0 --no-ui --log-verbosity 3 \
  >"$OUT/server.log" 2>&1 &
PID=$!
for _ in $(seq 1 900); do
  curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 "$PID" 2>/dev/null || { tail -100 "$OUT/server.log" >&2; exit 73; }
  sleep 1
done
curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null
python3 /home/edwin/.ralph/mtp-dynamic-controller-work/scripts/mtp-dynamic-concurrent-bench.py \
  --url "http://127.0.0.1:$PORT" --parallel "$PARALLEL" --n-predict "$N_PREDICT" \
  --prompt-kind "$PROMPT_KIND" --output "$OUT/result.json" | tee "$OUT/stdout"
grep -E "MTPCTRL|Q2CYCLE|disabling HIP graphs|MTP sidecar active" "$OUT/server.log" >"$OUT/important.log" || true
