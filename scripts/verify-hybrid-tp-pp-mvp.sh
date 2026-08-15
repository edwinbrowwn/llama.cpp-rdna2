#!/usr/bin/env bash
# Fresh-shell, external-monitor-rerunnable verifier for the hybrid TP×PP MVP.
set -euo pipefail

ROOT=${ROOT:-/home/edwin/llama.cpp-rdna2-hybrid-tp-pp}
EVIDENCE_ROOT=${EVIDENCE_ROOT:-/home/edwin/hybrid-tp-pp-evidence}
MODEL=${MODEL:-/home/edwin/models/qwen38-27b-q4s8/unsloth-q8/Qwen3.8-27B-Q8_0.gguf}
DRAFT=${DRAFT:-/home/edwin/models/qwen38-27b-q4s8/draft-q4/Qwen3.8-27B-MTP-Draft-Q4_0.gguf}
RUN_ROOT=${RUN_ROOT:-$EVIDENCE_ROOT/final-verifier-runs/$(date -u +%Y%m%dT%H%M%SZ)-$$}
JOBS=${JOBS:-24}
BASE_PORT=${BASE_PORT:-18600}
EXPECTED_BASE=db3eb285898d83883203bc2284307f4f7f544051
EXPECTED_MODEL_SHA=a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348
EXPECTED_DRAFT_SHA=a79d1b93250e68ed38dea083ca4185ddbc488861a9e2e0d51cd68b05a1ba4bb0

cd "$ROOT"
[[ "$(git branch --show-current)" == feat/hybrid-tp-pp-mvp ]] || { echo "wrong branch" >&2; exit 1; }
[[ -z "$(git status --porcelain=v1)" ]] || { echo "worktree is dirty" >&2; git status --short >&2; exit 1; }
[[ "$(git rev-parse HEAD)" == "$(git rev-parse fork/feat/hybrid-tp-pp-mvp)" ]] || { echo "feature branch differs from fork remote ref" >&2; exit 1; }
[[ "$(git -C /home/edwin/llama.cpp-rdna2 rev-parse master)" == "$EXPECTED_BASE" ]] || { echo "master worktree moved" >&2; exit 1; }
[[ "$(git -C /home/edwin/llama.cpp-rdna2 rev-parse fork/master)" == "$EXPECTED_BASE" ]] || { echo "fork/master moved during experiment" >&2; exit 1; }
[[ -r "$MODEL" && -r "$DRAFT" ]] || { echo "required model artifact missing" >&2; exit 1; }
if pgrep -x llama-server >/dev/null; then
    echo "an existing llama-server would invalidate final verification" >&2
    pgrep -a -x llama-server >&2
    exit 1
fi

mkdir -p "$RUN_ROOT"
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HSA_NO_SCRATCH_RECLAIM=1
export GGML_CUDA_DISABLE_GRAPHS=1
export GGML_CUDA_ALLREDUCE=nccl
export GGML_META_COPY_DEBUG=1
export GGML_SCHED_TRACE=0
export GGML_TP_SHARDED_OUTPUT=0
export LD_LIBRARY_PATH="$ROOT/build/bin:/opt/rocm/core-7.14/lib"
ulimit -s 8192

{
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "root=$ROOT"
    echo "evidence_root=$EVIDENCE_ROOT"
    echo "run_root=$RUN_ROOT"
    echo "feature_head=$(git rev-parse HEAD)"
    echo "feature_remote=$(git rev-parse fork/feat/hybrid-tp-pp-mvp)"
    echo "master=$(git -C /home/edwin/llama.cpp-rdna2 rev-parse master)"
    echo "master_remote=$(git -C /home/edwin/llama.cpp-rdna2 rev-parse fork/master)"
    env | grep -E '^(HSA_|GGML_|LD_LIBRARY_PATH=)' | sort
} >"$RUN_ROOT/manifest.txt"

verify_manifest() {
    local directory=$1
    [[ -r "$directory/SHA256SUMS" ]] || { echo "missing manifest: $directory/SHA256SUMS" >&2; return 1; }
    (cd "$directory" && sha256sum -c SHA256SUMS >/dev/null)
    echo "evidence_ok=$directory" >>"$RUN_ROOT/manifest.txt"
}

# Core correctness, timeline, state, numerical, and performance artifacts.
for directory in \
    "$EVIDENCE_ROOT/meta-async-copy" \
    "$EVIDENCE_ROOT/device-timeline" \
    "$EVIDENCE_ROOT/state-mutation" \
    "$EVIDENCE_ROOT/mtp-state-stress" \
    "$EVIDENCE_ROOT/legacy-regressions" \
    "$EVIDENCE_ROOT/numerical-equivalence" \
    "$EVIDENCE_ROOT/performance-mvp/pp512-tg128-c1c2-r3-68fb791c6" \
    "$EVIDENCE_ROOT/performance-mvp/pp512-tg128-c4c8-r3-4950a1605" \
    "$EVIDENCE_ROOT/performance-mvp/boundary-sweep-pp512-tg128-c1c2c4-r3-4950a1605" \
    "$EVIDENCE_ROOT/performance-mvp/pp2048-tg128-c4-r3-4950a1605" \
    "$EVIDENCE_ROOT/performance-mvp/pp4096-tg128-c4-r3-4950a1605" \
    "$EVIDENCE_ROOT/performance-mvp/pp512-tg512-c1c4-r3-447dc0ab7" \
    "$EVIDENCE_ROOT/performance-mvp/long-prefill-summary-447dc0ab7" \
    "$EVIDENCE_ROOT/performance-mvp/fixed-seed-validation-a81c9d663" \
    "$EVIDENCE_ROOT/performance-mvp/fixed-seed-hybrid-pp4096-c4-a81c9d663" \
    "$EVIDENCE_ROOT/performance-mvp/fixed-seed-legacy-tp2-pp4096-c4-a81c9d663"
do
    verify_manifest "$directory"
done

# Reanalyze immutable raw profiler traces into this run directory.
./scripts/analyze-hybrid-timeline.py \
    --trace "$EVIDENCE_ROOT/device-timeline/probe/hybrid-probe_results.json" \
    --log "$EVIDENCE_ROOT/device-timeline/probe/profiler.log" \
    --output "$RUN_ROOT/timeline-c1.json"
./scripts/analyze-hybrid-timeline.py \
    --trace "$EVIDENCE_ROOT/device-timeline/server-c4-profile/server-c4_results.json" \
    --log "$EVIDENCE_ROOT/device-timeline/server-c4-profile/profiler.log" \
    --output "$RUN_ROOT/timeline-c4.json"
python3 - "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
c1 = json.loads((root / "timeline-c1.json").read_text())
c4 = json.loads((root / "timeline-c4.json").read_text())
if c1["overlap_proven"]:
    raise SystemExit("C1 serial control unexpectedly reports overlap")
if not c4["overlap_proven"] or c4["positive_busy_overlap_pairs"] < 1:
    raise SystemExit("C4 device overlap is not proven")
print("FINAL_TIMELINE_REANALYSIS_OK")
PY

# Regenerate exact build metadata and compile current clean HEAD.
cmake -S . -B build >"$RUN_ROOT/cmake-configure.log" 2>&1
cmake --build build -j "$JOBS" \
    --target llama-server test-arg-parser test-qwen35moe-mmq-config \
             test-tensor-split test-meta-split test-parallel-topology \
    >"$RUN_ROOT/cmake-build.log" 2>&1
build_version=$(./build/bin/llama-server --version 2>&1 | head -1)
short_head=$(git rev-parse --short=9 HEAD)
[[ "$build_version" == *"($short_head)"* ]] || { echo "binary/source version mismatch: $build_version" >&2; exit 1; }
echo "server_$build_version" >>"$RUN_ROOT/manifest.txt"

ctest --test-dir build --output-on-failure \
    -R 'test-(arg-parser|qwen35moe-mmq-config|tensor-split|meta-split|parallel-topology)$' \
    | tee "$RUN_ROOT/ctest.log"

# Fresh bounded five-topology runtime gate. Hybrid MTP equality is required for
# this qualified short workload; TP4 output differences are audit-only.
OUT="$RUN_ROOT/runtime-matrix" \
PP_SPLIT=1,1 \
SEED_BASE=20260815 \
REQUIRE_MTP_OUTPUT_EQUIVALENCE=1 \
CONCURRENCY=2 \
PARALLEL=2 \
PROMPT_TOKENS=512 \
N_PREDICT=32 \
REPEATS=1 \
CTX_PER_SEQ=1024 \
BATCH=1024 \
UBATCH=512 \
BASE_PORT="$BASE_PORT" \
    ./scripts/run-hybrid-performance-matrix.sh | tee "$RUN_ROOT/runtime-matrix.stdout"

# The runtime manifest hashes both immutable models. Check the expected values.
grep -F "$EXPECTED_MODEL_SHA  $MODEL" "$RUN_ROOT/runtime-matrix/manifest.txt" >/dev/null
grep -F "$EXPECTED_DRAFT_SHA  $DRAFT" "$RUN_ROOT/runtime-matrix/manifest.txt" >/dev/null

[[ -z "$(git status --porcelain=v1)" ]] || { echo "verification dirtied worktree" >&2; exit 1; }
[[ "$(git rev-parse HEAD)" == "$(git rev-parse fork/feat/hybrid-tp-pp-mvp)" ]] || { echo "feature ref changed during verification" >&2; exit 1; }
! pgrep -x llama-server >/dev/null || { echo "server remained after verification" >&2; exit 1; }
for path in /sys/class/drm/card[0-9]/device/mem_info_vram_used; do
    used=$(<"$path")
    (( used < 67108864 )) || { echo "GPU VRAM did not return to baseline: $path=$used" >&2; exit 1; }
done

echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$RUN_ROOT/manifest.txt"
find "$RUN_ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >"$RUN_ROOT/SHA256SUMS"
(cd "$RUN_ROOT" && sha256sum -c SHA256SUMS >/dev/null)

echo "FINAL_RUN_ROOT=$RUN_ROOT"
echo "HYBRID_TP_PP_MVP_FINAL_VERIFY_OK"
