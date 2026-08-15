#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
OUT=${OUT:?set OUT to a fresh output directory}
MODEL=${MODEL:-/home/edwin/models/qwen38-27b-q4s8/unsloth-q8/Qwen3.8-27B-Q8_0.gguf}
EVIDENCE=${EVIDENCE:-/home/edwin/hybrid-tp-pp-evidence/tp2-pp2-beat-tp4-q8}
EXPECTED_MODEL_SHA=a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348
EXPECTED_MASTER=db3eb285898d83883203bc2284307f4f7f544051

[[ ! -e "$OUT" ]] || { echo "refusing existing OUT: $OUT" >&2; exit 2; }
mkdir -p "$OUT"
exec > >(tee "$OUT/verifier.log") 2>&1

cd "$ROOT"
[[ $(git branch --show-current) == perf/tp2-pp2-beat-tp4-q8 ]]
[[ -z $(git status --porcelain) ]]
[[ $(git rev-parse HEAD) == $(git rev-parse '@{upstream}') ]]
[[ $(git -C /home/edwin/llama.cpp-rdna2 rev-parse master) == "$EXPECTED_MASTER" ]]
[[ $(git rev-parse fork/master) == "$EXPECTED_MASTER" ]]
[[ $(sha256sum "$MODEL" | awk '{print $1}') == "$EXPECTED_MODEL_SHA" ]]
! pgrep -x llama-server >/dev/null

cat >"$OUT/source-state.txt" <<EOF
head=$(git rev-parse HEAD)
upstream=$(git rev-parse '@{upstream}')
master=$(git -C /home/edwin/llama.cpp-rdna2 rev-parse master)
model=$MODEL
model_sha256=$EXPECTED_MODEL_SHA
EOF

for rel in \
    iter1-server-pp512-tg128-c1c4 \
    iter2-device-profiles \
    iter4-two-plan-device-profile \
    iter4-c8-ub4-probe \
    iter6-c16-pp512-tg128 \
    iter6-c16-boundary-sweep \
    iter7-backend-sampling-c4 \
    iter7-optimistic-lower-bound; do
    test -s "$EVIDENCE/$rel/SHA256SUMS"
    sha256sum -c "$EVIDENCE/$rel/SHA256SUMS" >>"$OUT/evidence-manifests.log"
done

python3 - "$EVIDENCE" "$OUT/evidence-audit.json" <<'PY'
import json, pathlib, sys
E = pathlib.Path(sys.argv[1])
out_path = pathlib.Path(sys.argv[2])
base = json.load(open(E/'iter1-server-pp512-tg128-c1c4/comparison.json'))
c4 = base['hybrid_vs_tp4_percent']['4']
assert c4['aggregate_prompt_tokens_per_second_window'] > 0
assert c4['aggregate_generation_tokens_per_second_window'] < -10

timeline = json.load(open(E/'iter2-device-profiles/hybrid/timeline-summary.json'))
assert timeline['assigned_kernel_records'] == 185900
assert timeline['positive_busy_overlap_pairs'] == 0
assert timeline['stage0_to_previous_stage1_pairs'] == 40

critical = json.load(open(E/'iter2-device-profiles/hybrid/critical-path.json'))
assert critical['selected_dispatch_count'] == 30

ov = json.load(open(E/'iter4-two-plan-device-profile/overlap-summary.json'))
assert ov['positive'] == 7 and ov['overlap_ms'] > 17

output = json.load(open(E/'iter4-c8-ub4-probe/output-comparison.json'))
assert len(output) == 16 and not any(row['equal'] for row in output)

c16 = json.load(open(E/'iter6-c16-pp512-tg128/comparison.json'))
assert c16['hybrid_vs_tp4_percent']['aggregate_prompt_tokens_per_second_window'] > 0
assert c16['hybrid_vs_tp4_percent']['aggregate_generation_tokens_per_second_window'] < -10

boundary = json.load(open(E/'iter6-c16-boundary-sweep/comparison.json'))
assert boundary['best_hybrid_vs_tp4_percent']['aggregate_generation_tokens_per_second_window'] < -10

bs = json.load(open(E/'iter7-backend-sampling-c4/comparison.json'))
assert bs['hybrid_vs_tp4_percent']['aggregate_prompt_tokens_per_second_window'] > 0
assert bs['hybrid_vs_tp4_percent']['aggregate_generation_tokens_per_second_window'] < 0

bound = json.load(open(E/'iter7-optimistic-lower-bound/analysis.json'))
optimistic = bound['optimistic_projections']['also_remove_entire_hybrid_interdispatch_gap']
assert optimistic['vs_tp4_percent'] < -8

summary = {
    'schema_version': 1,
    'baseline_c4_percent': c4,
    'stable_overlap_pairs': timeline['positive_busy_overlap_pairs'],
    'experimental_overlap': ov,
    'experimental_equal_outputs': sum(row['equal'] for row in output),
    'c16_percent': c16['hybrid_vs_tp4_percent'],
    'backend_sampling_percent': bs['hybrid_vs_tp4_percent'],
    'optimistic_bound': optimistic,
}
out_path.write_text(json.dumps(summary, indent=2) + '\n')
print('TP2_PP2_Q8_EVIDENCE_AUDIT_OK')
PY

cmake --build build -j "${JOBS:-24}" --target \
    llama-server test-arg-parser test-qwen35moe-mmq-config \
    test-tensor-split test-meta-split test-parallel-topology
ctest --test-dir build --output-on-failure \
    -R 'test-(arg-parser|qwen35moe-mmq-config|tensor-split|meta-split|parallel-topology)$' \
    | tee "$OUT/ctest.log"
./build/bin/llama-server --version >"$OUT/build-version.txt" 2>&1

export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HSA_NO_SCRATCH_RECLAIM=1
export GGML_CUDA_DISABLE_GRAPHS=1
export GGML_CUDA_ALLREDUCE=nccl
export GGML_META_COPY_DEBUG=1
export GGML_SCHED_TRACE=0
export GGML_META_TRACE=0
export LD_LIBRARY_PATH="$ROOT/build/bin:/opt/rocm/core-7.14/lib"
unset LLAMA_GRAPH_REUSE_DISABLE LLAMA_HYBRID_PIPELINED_REUSE GGML_META_GRAPH_CACHE
unset GGML_TP_SHARDED_OUTPUT GGML_TP_VOCAB_SHARDED_OUTPUT
unset GGML_HIP_GFX1030_NATIVE GGML_HIP_GFX1030_GDN_SIBLING_FUSION GGML_HIP_GFX1030_Q8_CACHE
unset GGML_HIP_RDNA2_BF16_HIDDEN_ALLREDUCE
ulimit -s 8192

COMMON=(
    -lv 5 -m "$MODEL" -c 512 -b 128 -ub 128 --parallel 2
    --no-warmup --ctx-checkpoints 0 -ngl all --split-mode tensor
    --fit off -fa on --spec-type none -dev ROCm0,ROCm1,ROCm2,ROCm3
)
mkdir -p "$OUT/hybrid" "$OUT/tp4"
timeout 1800s ./scripts/benchmark-hybrid-server.py \
    --label hybrid --output-dir "$OUT/hybrid" --port "${HYBRID_PORT:-18870}" \
    --concurrency 2 --prompt-tokens 64 --n-predict 8 --repeats 1 \
    --seed-base 20260815 --require-meta-zero-fallback -- \
    ./build/bin/llama-server "${COMMON[@]}" -ts 1,1 \
    --tp-size 2 --pp-size 2 --pp-split 1,1
timeout 1800s ./scripts/benchmark-hybrid-server.py \
    --label tp4 --output-dir "$OUT/tp4" --port "${TP4_PORT:-18871}" \
    --concurrency 2 --prompt-tokens 64 --n-predict 8 --repeats 1 \
    --seed-base 20260815 -- \
    ./build/bin/llama-server "${COMMON[@]}" -ts 1,1,1,1

python3 - "$OUT" <<'PY'
import json, pathlib, sys
O = pathlib.Path(sys.argv[1])
h = json.load(open(O/'hybrid/hybrid.json'))
t = json.load(open(O/'tp4/tp4.json'))
assert h['pipeline_parallel_enabled'] is True
assert t['pipeline_parallel_enabled'] is False
assert h['communicator_sizes'] == [2, 2]
assert t['communicator_sizes'] == [4]
assert h['meta_fallback_gate']['passed'] is True
active = [row for row in h['meta_copy_telemetry'] if row['meta_attempts'] > 0]
assert active and all(row['success'] == row['meta_attempts'] and row['meta_fallback'] == 0 for row in active)
hr = h['groups'][0]['requests']
tr = t['groups'][0]['requests']
assert len(hr) == len(tr) == 2
assert [r['tokens'] for r in hr] == [r['tokens'] for r in tr]
summary = {
    'schema_version': 1,
    'hybrid_communicators': h['communicator_sizes'],
    'tp4_communicators': t['communicator_sizes'],
    'hybrid_pipeline': h['pipeline_parallel_enabled'],
    'tp4_pipeline': t['pipeline_parallel_enabled'],
    'bounded_output_equal': True,
    'hybrid_meta_rows': active,
    'hybrid_generation_tps': h['groups'][0]['aggregate_generation_tokens_per_second_window'],
    'tp4_generation_tps': t['groups'][0]['aggregate_generation_tokens_per_second_window'],
}
(O/'runtime-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print('TP2_PP2_Q8_RUNTIME_GATE_OK')
PY

! pgrep -x llama-server >/dev/null
for f in /sys/class/drm/card[0-9]/device/mem_info_vram_used; do
    [[ $(cat "$f") -lt 67108864 ]]
done
find "$OUT" -type f ! -name SHA256SUMS ! -name verifier.log -print0 | sort -z | xargs -0 sha256sum >"$OUT/SHA256SUMS"
echo TP2_PP2_VS_TP4_Q8_FINAL_VERIFY_OK
