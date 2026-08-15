#!/usr/bin/env bash
# Launch the five matched server topologies for one PP/TG/concurrency workload.
set -euo pipefail

ROOT=${ROOT:-/home/edwin/llama.cpp-rdna2-hybrid-tp-pp}
OUT=${OUT:?set OUT to a new evidence directory}
MODEL=${MODEL:-/home/edwin/models/qwen38-27b-q4s8/unsloth-q8/Qwen3.8-27B-Q8_0.gguf}
DRAFT=${DRAFT:-/home/edwin/models/qwen38-27b-q4s8/draft-q4/Qwen3.8-27B-MTP-Draft-Q4_0.gguf}
CONCURRENCY=${CONCURRENCY:-1,2}
PARALLEL=${PARALLEL:-2}
PROMPT_TOKENS=${PROMPT_TOKENS:-512}
N_PREDICT=${N_PREDICT:-128}
REPEATS=${REPEATS:-2}
CTX_PER_SEQ=${CTX_PER_SEQ:-2048}
BATCH=${BATCH:-1024}
UBATCH=${UBATCH:-512}
BASE_PORT=${BASE_PORT:-18120}
CASES=${CASES:-tp2-pp1-nospec,tp4-pp1-nospec,tp4-pp1-external-mtp,tp2xpp2-nospec,tp2xpp2-external-mtp}

cd "$ROOT"
[[ -x build/bin/llama-server ]] || { echo "missing build/bin/llama-server" >&2; exit 1; }
[[ -r "$MODEL" ]] || { echo "missing target model: $MODEL" >&2; exit 1; }
[[ -r "$DRAFT" ]] || { echo "missing draft model: $DRAFT" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite existing output: $OUT" >&2; exit 1; }
if pgrep -x llama-server >/dev/null; then
    echo "an existing llama-server process would invalidate the benchmark" >&2
    pgrep -a -x llama-server >&2
    exit 1
fi
max_concurrency=$(tr ',' '\n' <<<"$CONCURRENCY" | sort -nr | head -1)
(( PARALLEL >= max_concurrency )) || { echo "PARALLEL must be >= maximum concurrency" >&2; exit 1; }
(( CTX_PER_SEQ >= PROMPT_TOKENS + N_PREDICT )) || { echo "CTX_PER_SEQ is too small" >&2; exit 1; }

mkdir -p "$OUT"
export HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-10.3.0}
export HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-1}
export GGML_CUDA_DISABLE_GRAPHS=${GGML_CUDA_DISABLE_GRAPHS:-1}
export GGML_CUDA_ALLREDUCE=${GGML_CUDA_ALLREDUCE:-nccl}
export GGML_META_COPY_DEBUG=1
export GGML_SCHED_TRACE=0
export GGML_TP_SHARDED_OUTPUT=0
export LD_LIBRARY_PATH="$ROOT/build/bin:/opt/rocm/core-7.14/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ulimit -s 8192

ctx_total=$(( CTX_PER_SEQ * PARALLEL ))
{
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "root=$ROOT"
    echo "out=$OUT"
    echo "model=$MODEL"
    echo "draft=$DRAFT"
    echo "concurrency=$CONCURRENCY"
    echo "parallel=$PARALLEL"
    echo "prompt_tokens=$PROMPT_TOKENS"
    echo "n_predict=$N_PREDICT"
    echo "repeats=$REPEATS"
    echo "ctx_per_seq=$CTX_PER_SEQ"
    echo "ctx_total=$ctx_total"
    echo "batch=$BATCH"
    echo "ubatch=$UBATCH"
    echo "cases=$CASES"
    echo "git_head=$(git rev-parse HEAD)"
    echo "git_status=$(git status --porcelain=v1 | tr '\n' ';')"
    build/bin/llama-server --version 2>&1 | sed 's/^/server_/'
    env | grep -E '^(HSA_|GGML_|LD_LIBRARY_PATH=)' | sort
    sha256sum "$MODEL" "$DRAFT"
} >"$OUT/manifest.txt"

contains_case() {
    [[ ",$CASES," == *",$1,"* ]]
}

common=(
    # The common logger maps low-level GGML Meta INFO records at verbosity 5.
    # This retains the required copy summary and MTP acceptance telemetry.
    -lv 5 -m "$MODEL" -c "$ctx_total" -b "$BATCH" -ub "$UBATCH"
    --parallel "$PARALLEL" --no-warmup --ctx-checkpoints 0
    -ngl all --split-mode tensor --fit off -fa on
)
mtp=(
    --spec-type draft-mtp --model-draft "$DRAFT" --spec-draft-n-max 2 --spec-draft-p-min 0
    --gpu-layers-draft all -devd ROCm2,ROCm3
)

run_case() {
    local label=$1
    local port=$2
    local require_meta=$3
    shift 3
    local case_out="$OUT/$label"
    local gate=()
    [[ "$require_meta" == 1 ]] && gate+=(--require-meta-zero-fallback)
    mkdir -p "$case_out"
    echo "running $label on port $port"
    timeout 3600s ./scripts/benchmark-hybrid-server.py \
        --label "$label" --output-dir "$case_out" --port "$port" \
        --concurrency "$CONCURRENCY" --prompt-tokens "$PROMPT_TOKENS" \
        --n-predict "$N_PREDICT" --repeats "$REPEATS" "${gate[@]}" -- \
        ./build/bin/llama-server "${common[@]}" "$@" \
        2>&1 | tee "$case_out/harness.stdout"
    sha256sum "$case_out/$label.json" "$case_out/$label.server.log" >"$case_out/SHA256SUMS"
}

contains_case tp2-pp1-nospec && run_case tp2-pp1-nospec "$BASE_PORT" 0 \
    -dev ROCm0,ROCm1 -ts 1,1 --spec-type none
contains_case tp4-pp1-nospec && run_case tp4-pp1-nospec "$((BASE_PORT + 1))" 0 \
    -dev ROCm0,ROCm1,ROCm2,ROCm3 -ts 1,1,1,1 --spec-type none
contains_case tp4-pp1-external-mtp && run_case tp4-pp1-external-mtp "$((BASE_PORT + 2))" 0 \
    -dev ROCm0,ROCm1,ROCm2,ROCm3 -ts 1,1,1,1 "${mtp[@]}"
contains_case tp2xpp2-nospec && run_case tp2xpp2-nospec "$((BASE_PORT + 3))" 1 \
    -dev ROCm0,ROCm1,ROCm2,ROCm3 -ts 1,1 --tp-size 2 --pp-size 2 --pp-split 1,1 --spec-type none
contains_case tp2xpp2-external-mtp && run_case tp2xpp2-external-mtp "$((BASE_PORT + 4))" 1 \
    -dev ROCm0,ROCm1,ROCm2,ROCm3 -ts 1,1 --tp-size 2 --pp-size 2 --pp-split 1,1 "${mtp[@]}"

./scripts/summarize-hybrid-performance.py --output-dir "$OUT/summary" "$OUT"
python3 - "$OUT" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
expect = {
    "tp2-pp1-nospec": ([2], False),
    "tp4-pp1-nospec": ([4], False),
    "tp4-pp1-external-mtp": ([4, 2], False),
    "tp2xpp2-nospec": ([2, 2], True),
    "tp2xpp2-external-mtp": ([2, 2, 2], True),
}
for path in root.glob("*/*.json"):
    data = json.loads(path.read_text())
    label = data.get("label")
    if label not in expect:
        continue
    communicators, pipeline = expect[label]
    if data["communicator_sizes"] != communicators:
        raise SystemExit(f"{label}: communicator sizes {data['communicator_sizes']} != {communicators}")
    if data["pipeline_parallel_enabled"] is not pipeline:
        raise SystemExit(f"{label}: pipeline flag {data['pipeline_parallel_enabled']} != {pipeline}")
    if any(request["truncated"] for group in data["groups"] for request in group["requests"]):
        raise SystemExit(f"{label}: a request was truncated")
print("HYBRID_MATRIX_STRUCTURE_OK")
PY
find "$OUT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >"$OUT/SHA256SUMS"
echo "HYBRID_PERFORMANCE_MATRIX_OK"
