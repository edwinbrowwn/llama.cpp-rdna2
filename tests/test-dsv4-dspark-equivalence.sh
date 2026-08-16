#!/usr/bin/env bash
# Model-dependent DSpark correctness check for the RDNA2 DSV4 branch.
# It deliberately uses ubatch=1: the branch refuses unsafe batched AMD
# DFlash/DSpark verification rather than silently changing target behavior.
set -Eeuo pipefail

: "${DSV4_MODEL:?set DSV4_MODEL to the target GGUF shard}"
: "${DSV4_DRAFT_MODEL:?set DSV4_DRAFT_MODEL to the DSpark GGUF}"
[[ -f "$DSV4_MODEL" && -f "$DSV4_DRAFT_MODEL" ]]

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SERVER=${DSV4_SERVER:-$ROOT_DIR/build/bin/llama-server}
MODE=${DSV4_SPLIT_MODE:-layer}
TENSOR_SPLIT=${DSV4_TENSOR_SPLIT:-1,1,1,1}
N_MAX=${DSV4_N_MAX:-3}
N_PREDICT=${DSV4_N_PREDICT:-32}
PORT=${DSV4_PORT:-18180}
BATCH=${DSV4_BATCH_SIZE:-32}
UBATCH=${DSV4_UBATCH_SIZE:-1}
CTX=${DSV4_CTX_SIZE:-4096}
[[ -x "$SERVER" ]]
[[ "$MODE" == layer || "$MODE" == tensor ]]
[[ "$N_MAX" =~ ^[1-9][0-9]*$ && "$N_PREDICT" =~ ^[1-9][0-9]*$ ]]
[[ "$BATCH" =~ ^[1-9][0-9]*$ && "$UBATCH" =~ ^[1-9][0-9]*$ && "$UBATCH" -le "$BATCH" ]]

TMP=$(mktemp -d "${TMPDIR:-/tmp}/dsv4-dspark-equivalence.XXXXXX")
PID=""
cleanup() {
    if [[ -n "$PID" ]]; then
        kill "$PID" 2>/dev/null || true
        wait "$PID" 2>/dev/null || true
    fi
    rm -rf "$TMP"
}
trap cleanup EXIT INT TERM

REQUEST=$(python3 - "$N_PREDICT" <<'PY'
import json, sys
print(json.dumps({
    "prompt": "Explain why a shared prefix improves inference throughput.",
    "n_predict": int(sys.argv[1]),
    "seed": 123,
    "temperature": 0,
    "cache_prompt": False,
    "return_tokens": True,
}))
PY
)

run_server() {
    local label=$1 draft=$2 port=$3
    local log="$TMP/$label.log" response="$TMP/$label.json"
    local -a extra=(--model "$DSV4_MODEL")
    if [[ "$draft" == 1 ]]; then
        extra+=(--spec-draft-model "$DSV4_DRAFT_MODEL" --spec-type draft-dspark --spec-draft-n-max "$N_MAX")
    fi
    "$SERVER" -lv 1 "${extra[@]}" \
        --host 127.0.0.1 --port "$port" --ctx-size "$CTX" \
        --batch-size "$BATCH" --ubatch-size "$UBATCH" --parallel 1 \
        --seed 123 --temp 0 --flash-attn auto --split-mode "$MODE" \
        --tensor-split "$TENSOR_SPLIT" --ctx-checkpoints 0 --no-warmup \
        >"$log" 2>&1 &
    PID=$!
    for _ in $(seq 1 "${DSV4_STARTUP_RETRIES:-2400}"); do
        if curl -sf --max-time 2 "http://127.0.0.1:$port/props" >/dev/null; then
            break
        fi
        kill -0 "$PID" 2>/dev/null || { cat "$log" >&2; return 1; }
        sleep 1
    done
    curl -sf --max-time "${DSV4_REQUEST_TIMEOUT:-2400}" \
        -H 'Content-Type: application/json' --data "$REQUEST" \
        -o "$response" "http://127.0.0.1:$port/completion"
    kill "$PID" 2>/dev/null || true
    wait "$PID" 2>/dev/null || true
    PID=""
}

run_server nodraft 0 "$PORT"
run_server dspark 1 "$((PORT + 1))"

python3 - "$TMP/nodraft.json" "$TMP/dspark.json" <<'PY'
import json, sys
no = json.load(open(sys.argv[1]))
draft = json.load(open(sys.argv[2]))
if no.get("tokens") != draft.get("tokens"):
    for i, (a, b) in enumerate(zip(no.get("tokens", []), draft.get("tokens", []))):
        if a != b:
            raise SystemExit(f"first token mismatch at {i}: no-draft={a} dspark={b}")
    raise SystemExit(f"token length mismatch: no-draft={len(no.get('tokens', []))} dspark={len(draft.get('tokens', []))}")
if draft.get("timings", {}).get("draft_attempts", 0) <= 0:
    raise SystemExit("DSpark did not execute any draft attempts")
print("DSpark target-token equivalence passed")
print("tokens:", len(draft["tokens"]), "draft attempts:", draft["timings"].get("draft_attempts"), "accepted:", draft["timings"].get("draft_n_accepted"))
PY
