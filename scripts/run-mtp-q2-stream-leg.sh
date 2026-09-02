#!/usr/bin/env bash
set -Eeuo pipefail

LABEL=${LABEL:?set LABEL}
SERVER_BIN=${SERVER_BIN:-/home/edwin/.ralph/unified-rdna2-rdna3-build-gfx1030/bin/llama-server}
PORT=${PORT:-18210}
STACK=${STACK:-mtp}
TEMPERATURE=${TEMPERATURE:-1.0}
N_PREDICT=${N_PREDICT:-256}
SEED=${SEED:-27742}
RUN_ROOT=${RUN_ROOT:-/home/edwin/.ralph/mtp-q2-smooth-artifacts}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}
PROMPT_FILE=${PROMPT_FILE:-}
MODEL=${MODEL:-/home/edwin/models/qwen38-27b-magicquant-iq4-xs/Qwen3.8-27B-Quark-MXFP4-MQ-IQ4_XS_1-Generic.gguf}
MMPROJ=${MMPROJ:-/home/edwin/models/qwen38-27b-q4s8/autoround-q4-fast/mmproj-model.gguf}
ROCM=${ROCM:-/opt/rocm/core-7.14}
LOG_VERBOSITY=${LOG_VERBOSITY:-3}
CYCLE_TRACE=${CYCLE_TRACE:-1}
SPEC_DEPTH=${SPEC_DEPTH:-4}
NGRAM_N=${NGRAM_N:-12}
NGRAM_M=${NGRAM_M:-48}
REQUEST_SPEC_N_MAX=${REQUEST_SPEC_N_MAX:-}

[[ $PORT =~ ^[0-9]+$ && $PORT -gt 0 && $PORT -lt 65535 && $PORT -ne 8080 ]]
[[ $N_PREDICT =~ ^[1-9][0-9]*$ ]]
[[ $SPEC_DEPTH =~ ^[1-8]$ ]]
[[ $NGRAM_N =~ ^[1-9][0-9]*$ && $NGRAM_M =~ ^[1-9][0-9]*$ ]]
[[ $STACK == mtp || $STACK == stacked || $STACK == k4v ]]
[[ -x $SERVER_BIN && -s $MODEL && -s $MMPROJ ]]
BIN_DIR=$(dirname "$SERVER_BIN")
[[ -x $BIN_DIR/spec_hip_sidecar.so || -f $BIN_DIR/spec_hip_sidecar.so ]]
OUT="$RUN_ROOT/$RUN_ID/$LABEL"
mkdir -p "$OUT"
LOG="$OUT/server.log"
PID=

cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    set +e
    if [[ -n ${PID:-} ]] && kill -0 "$PID" 2>/dev/null; then
        kill -TERM "$PID" 2>/dev/null || true
        for _ in $(seq 1 80); do
            kill -0 "$PID" 2>/dev/null || break
            sleep 0.25
        done
        kill -KILL "$PID" 2>/dev/null || true
        wait "$PID" 2>/dev/null || true
    fi
    if fuser /dev/kfd >/dev/null 2>&1; then
        echo "post_idle=no" >&2
        fuser -v /dev/kfd >&2 || true
        [[ $rc -ne 0 ]] || rc=76
    else
        echo "post_idle=yes" | tee "$OUT/post-idle.txt"
    fi
    exit "$rc"
}
trap cleanup EXIT INT TERM

if fuser /dev/kfd >/dev/null 2>&1; then
    echo "pre_idle=no" >&2; fuser -v /dev/kfd >&2 || true; exit 75
fi
if ss -ltn "sport = :$PORT" | grep -q LISTEN; then
    echo "port_busy=$PORT" >&2; exit 74
fi

case "$STACK" in
    mtp)     SPEC_ARGS=(--spec-type draft-mtp) ;;
    stacked) SPEC_ARGS=(--spec-type draft-mtp,ngram-map-k4v --spec-ngram-map-k4v-size-n "$NGRAM_N" --spec-ngram-map-k4v-size-m "$NGRAM_M") ;;
    k4v)     SPEC_ARGS=(--spec-type ngram-map-k4v --spec-ngram-map-k4v-size-n "$NGRAM_N" --spec-ngram-map-k4v-size-m "$NGRAM_M") ;;
esac

cat >"$OUT/config.txt" <<EOF
label=$LABEL
run_id=$RUN_ID
server=$SERVER_BIN
model=$MODEL
stack=$STACK
temperature=$TEMPERATURE
n_predict=$N_PREDICT
seed=$SEED
spec_depth=$SPEC_DEPTH
ngram_n=$NGRAM_N
ngram_m=$NGRAM_M
request_spec_n_max=$REQUEST_SPEC_N_MAX
port=$PORT
prompt_file=$PROMPT_FILE
EOF

echo "pre_idle=yes label=$LABEL stack=$STACK port=$PORT"
export PATH="$ROCM/bin:$ROCM/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$BIN_DIR:$ROCM/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export SPEC_SIDECAR=1
export HSA_NO_SCRATCH_RECLAIM=1
export GGML_HIP_RDNA2_AUTO=1
export GGML_HIP_SAFE_STATE_IO=1
export GGML_TP_SHARDED_OUTPUT=1
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HIP_VISIBLE_DEVICES=0,1,2,3
export LLAMA_SPEC_CYCLE_TRACE=$CYCLE_TRACE

"$SERVER_BIN" \
  -m "$MODEL" -ngl all --split-mode tensor --tensor-split 1,1,1,1 \
  --device ROCm0,ROCm1,ROCm2,ROCm3 --flash-attn on --ctx-size 262144 \
  --batch-size 8192 --ubatch-size 4096 --host 127.0.0.1 --port "$PORT" \
  --temp "$TEMPERATURE" --top-p 0.95 --top-k 20 --min-p 0 \
  --presence-penalty 0 --repeat-penalty 1.0 --metrics \
  --reasoning-effort xhigh --reasoning-preserve \
  "${SPEC_ARGS[@]}" --spec-draft-n-max "$SPEC_DEPTH" --spec-draft-p-min 0 \
  --parallel 1 --spec-draft-ubatch-size 4096 --cache-ram 65535 \
  --mmproj "$MMPROJ" --ctx-checkpoints 0 --no-ui --log-verbosity "$LOG_VERBOSITY" \
  >"$LOG" 2>&1 &
PID=$!

ready=0
for _ in $(seq 1 900); do
    if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        ready=1; break
    fi
    if ! kill -0 "$PID" 2>/dev/null; then
        echo server_exited >&2; tail -180 "$LOG" >&2 || true; exit 73
    fi
    sleep 1
done
[[ $ready == 1 ]] || { echo startup_timeout >&2; tail -180 "$LOG" >&2; exit 72; }

PY_ARGS=(
  --url "http://127.0.0.1:$PORT"
  --out "$OUT/stream"
  --n-predict "$N_PREDICT"
  --seed "$SEED"
  --temperature "$TEMPERATURE"
)
[[ -z $PROMPT_FILE ]] || PY_ARGS+=(--prompt-file "$PROMPT_FILE")
[[ -z $REQUEST_SPEC_N_MAX ]] || PY_ARGS+=(--spec-n-max "$REQUEST_SPEC_N_MAX")
python3 /home/edwin/.ralph/mtp-q2-smooth-work/scripts/mtp-q2-stream-latency.py "${PY_ARGS[@]}" \
  | tee "$OUT/stream.stdout"

# Preserve concise lifecycle evidence and any experiment-only per-cycle lines.
grep -E "sidecar-only draft selected|MTP sidecar active|capping automatic|sidecar ngram verification|draft acceptance|Q2CYCLE|target-only|catch-up.*failed|HIP fail|invalid argument|GGML_ASSERT" \
  "$LOG" >"$OUT/important.log" || true
if grep -q "Q2CYCLE " "$LOG"; then
    python3 /home/edwin/.ralph/mtp-q2-smooth-work/scripts/parse-mtp-q2-cycle-trace.py \
      --log "$LOG" --out "$OUT/cycles" | tee "$OUT/cycles.stdout"
fi
if grep -Eiq "HIP fail|invalid argument|GGML_ASSERT|segmentation fault|memory fault" "$LOG"; then
    echo runtime_error_marker >&2; exit 71
fi
sha256sum "$OUT/stream.summary.json" "$OUT/stream.tokens.json" "$LOG" >"$OUT/SHA256SUMS"
echo "artifact=$OUT"
