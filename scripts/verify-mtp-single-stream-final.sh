#!/usr/bin/env bash
set -Eeuo pipefail

WORK=${WORK:-/home/edwin/.ralph/mtp-single-stream-continuation-work}
BUILD=${BUILD:-/home/edwin/.ralph/mtp-single-stream-continuation-build-gfx1030}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/home/edwin/.ralph/mtp-single-stream-continuation-artifacts}
PRODUCTION=${PRODUCTION:-/home/edwin/llama.cpp-rdna2/build/bin/llama-server}
RUN_ID=${RUN_ID:-final-$(date -u +%Y%m%d-%H%M%S)}
PORT_BASE=${PORT_BASE:-18680}
OUT="$ARTIFACT_ROOT/$RUN_ID"
mkdir -p "$OUT"

[[ $PORT_BASE =~ ^[0-9]+$ && $PORT_BASE -ge 1024 && $((PORT_BASE + 7)) -lt 65535 ]]
for port in $(seq "$PORT_BASE" $((PORT_BASE + 7))); do
    [[ $port -ne 8080 ]]
    if ss -ltn "sport = :$port" | grep -q LISTEN; then
        echo "port busy: $port" >&2; exit 74
    fi
done
if fuser /dev/kfd >/dev/null 2>&1; then
    echo pre_idle=no >&2; fuser -v /dev/kfd >&2 || true; exit 75
fi
cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    if fuser /dev/kfd >/dev/null 2>&1; then
        echo post_idle=no >&2; fuser -v /dev/kfd >&2 || true
        [[ $rc -ne 0 ]] || rc=76
    else
        echo post_idle=yes
    fi
    exit "$rc"
}
trap cleanup EXIT INT TERM

echo "run_id=$RUN_ID"
echo "pre_idle=yes"

git -C "$WORK" diff --check
test -z "$(git -C "$WORK" status --short)"
BASE=0ace3850324be7af56fc40ed016559b767d7f8f9
git -C "$WORK" merge-base --is-ancestor "$BASE" HEAD
if grep -RInE "proposal_top_p|MTP_PROPOSAL_TOP|K4V_CONTINUATION|mtp_p1_sticky|MTPP1|shared_head_head_q2|MTP_HEAD_MODE" \
        "$WORK/common" "$WORK/tools/server" "$WORK/src/spec_sidecar/mtp" "$WORK/tests" >/dev/null 2>&1; then
    echo "rejected experiment source remains" >&2; exit 70
fi

echo "head=$(git -C "$WORK" rev-parse HEAD)" | tee "$OUT/provenance.txt"
echo "base=$BASE" | tee -a "$OUT/provenance.txt"
git -C "$WORK" diff --stat "$BASE" | tee "$OUT/diff-stat.txt"
PROD_BEFORE=$(sha256sum "$PRODUCTION" | awk '{print $1}')
echo "production_before=$PROD_BEFORE" | tee -a "$OUT/provenance.txt"

cmake --build "$BUILD" --target \
    llama-server llama-speculative-simple spec-sidecar-hip-mtp spec-sidecar-hip-qwen35moe-mtp \
    test-spec-sidecar-artifact test-speculative-sidecar-cap \
    test-speculative-backend-policy test-speculative-mtp-controller \
    -j 12 >"$OUT/build.log" 2>&1
"$BUILD/bin/llama-server" --version 2>&1 | tee "$OUT/version.txt"
ctest --test-dir "$BUILD" \
    -R "^(test-spec-sidecar-artifact|test-speculative-sidecar-cap|test-speculative-backend-policy|test-speculative-mtp-controller)$" \
    --output-on-failure | tee "$OUT/ctest.log"

run_leg() {
    local label=$1 port=$2 depth=$3 stack=$4 temp=$5 n_predict=$6 seed=$7 prompt=$8 request_cap=${9:-}
    env -u LLAMA_SPEC_MTP_NEURAL_DEPTH -u LLAMA_SPEC_MTP_DYNAMIC_DEPTH \
        -u GGML_CUDA_DISABLE_GRAPHS -u GGML_HIP_GFX1030_SPEC_GRAPHS \
        RUN_ROOT="$ARTIFACT_ROOT" RUN_ID="$RUN_ID" LABEL="$label" \
        SERVER_BIN="$BUILD/bin/llama-server" PORT="$port" STACK="$stack" \
        SPEC_DEPTH="$depth" NGRAM_N=12 NGRAM_M=48 TEMPERATURE="$temp" \
        N_PREDICT="$n_predict" SEED="$seed" PROMPT_FILE="$prompt" \
        REQUEST_SPEC_N_MAX="$request_cap" \
        "$WORK/scripts/run-mtp-q2-stream-leg.sh" >"$OUT/$label.run.log" 2>&1
}

ORDINARY="$WORK/scripts/prompts/ordinary.txt"
REPEAT="$WORK/scripts/prompts/repeat.txt"
run_leg standard-t0       "$PORT_BASE"       4 mtp     0.0 64  27742 "$ORDINARY"
run_leg standard-t1       "$((PORT_BASE+1))" 4 mtp     1.0 64  27812 "$ORDINARY"
run_leg repeat-t0-w4      "$((PORT_BASE+2))" 4 stacked 0.0 128 27742 "$REPEAT"
run_leg repeat-t0-w5      "$((PORT_BASE+3))" 5 stacked 0.0 128 27742 "$REPEAT"
run_leg repeat-t1-w4      "$((PORT_BASE+4))" 4 stacked 1.0 512 27812 "$REPEAT"
run_leg repeat-t1-w5      "$((PORT_BASE+5))" 5 stacked 1.0 512 27812 "$REPEAT"
run_leg ordinary-t1-w5    "$((PORT_BASE+6))" 5 stacked 1.0 256 27812 "$ORDINARY"
run_leg repeat-cap2-w5    "$((PORT_BASE+7))" 5 stacked 0.0 64  27742 "$REPEAT" 2

python3 - "$OUT" <<'PY' | tee "$OUT/validate.stdout"
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])

def need(value, message):
    if not value:
        raise SystemExit("FAIL: " + message)

def stream(label):
    return json.loads((out / label / "stream.summary.json").read_text())

def cycles(label):
    return json.loads((out / label / "cycles.summary.json").read_text())

def cycle_rows(label):
    return [json.loads(line) for line in (out / label / "cycles.jsonl").read_text().splitlines()]

def log(label):
    return (out / label / "server.log").read_text(errors="replace")

def result(label):
    s = stream(label)
    c = cycles(label)
    rows = cycle_rows(label)
    steady_rows = rows[2:]
    return {
        "tps": s["server_timings"]["predicted_per_second"],
        "draft": s["server_timings"]["draft_n"],
        "accepted": s["server_timings"]["draft_n_accepted"],
        "hash": s["token_sha256"],
        "sources": c["source_counts"],
        "proposed": c["proposed_histogram"],
        "cycles": c["cycles"],
        "first_reject_rate": c["first_reject_rate"],
        "cycle_p50_ms": c["steady"]["cycle_us"]["p50"] / 1000.0,
        "cycle_p95_ms": c["steady"]["cycle_us"]["p95"] / 1000.0,
        "cycle_p99_ms": c["steady"]["cycle_us"]["p99"] / 1000.0,
        "cycle_max_ms": c["steady"]["cycle_us"]["max"] / 1000.0,
        "warm_gt100": sum(row["cycle_us"] > 100000 for row in steady_rows),
        "warm_gt200": sum(row["cycle_us"] > 200000 for row in steady_rows),
        "sse": s["gap_ms"],
    }

labels = (
    "standard-t0", "standard-t1", "repeat-t0-w4", "repeat-t0-w5",
    "repeat-t1-w4", "repeat-t1-w5", "ordinary-t1-w5", "repeat-cap2-w5",
)
r = {label: result(label) for label in labels}

need((r["standard-t0"]["draft"], r["standard-t0"]["accepted"], r["standard-t0"]["hash"]) ==
     (105, 36, "8a8746899968363dc1d17c770e03aa2f739713bb2708632c298d016261972c37"),
     "standard temperature-zero parity changed")
need((r["standard-t1"]["draft"], r["standard-t1"]["accepted"], r["standard-t1"]["hash"]) ==
     (81, 42, "e35e3717ef97181cbf42992ad30fa5040b8062cdb39969076ca62ad5c0113e87"),
     "standard stochastic trajectory changed")
need("MTPCTRL" not in log("standard-t1"), "default automatic controller still floods info logs")
need(max(map(int, r["standard-t1"]["proposed"])) <= 4, "standard profile exceeded width four")

need(r["repeat-t0-w4"]["hash"] == r["repeat-t0-w5"]["hash"],
     "temperature-zero K4V width changed target output")
need(r["repeat-t1-w4"]["hash"] == r["repeat-t1-w5"]["hash"],
     "repeat stochastic target output diverged")
need(max(map(int, r["repeat-t1-w4"]["proposed"])) <= 4, "standard K4V exceeded four")
need(max(map(int, r["repeat-t1-w5"]["proposed"])) <= 5, "capacity-five K4V exceeded five")
need(r["repeat-t1-w5"]["sources"].get("ngram-map-k4v", 0) > 0 and
     r["repeat-t1-w5"]["sources"].get("draft-mtp", 0) > 0,
     "capacity-five run did not exercise both proposal sources")
mtp_widths = [row["proposed"] for row in cycle_rows("repeat-t1-w5") if row["source"] == "draft-mtp"]
k4v_widths = [row["proposed"] for row in cycle_rows("repeat-t1-w5") if row["source"] == "ngram-map-k4v"]
need(mtp_widths and max(mtp_widths) <= 4, "source-specific profile widened neural MTP")
need(k4v_widths and 5 in k4v_widths and max(k4v_widths) <= 5,
     "source-specific profile did not exercise bounded K4V width five")
need("using source-specific p1 widths: neural MTP=4, bounded K4V=5" in log("repeat-t1-w5"),
     "source-specific profile log absent")
need("capping automatic gfx1030 MTP+K4V cycles at 5" in log("repeat-t1-w5"),
     "automatic K4V cap-five log absent")
need("disabling HIP graphs for smooth gfx1030 TP4 MTP verification" not in log("repeat-t1-w5"),
     "automatic graph suppression is still active")

need(r["repeat-t1-w4"]["tps"] > 0.0 and r["repeat-t1-w5"]["tps"] > 0.0,
     "graph-on K4V throughput measurement is invalid")
need(r["repeat-t1-w5"]["warm_gt200"] <= 2,
     "capacity-five graph-on path has more than two warm cycles over 200 ms")
need(r["repeat-t1-w5"]["cycle_max_ms"] < 300.0,
     "capacity-five graph-on warm cycle maximum exceeded 300 ms")
need(r["repeat-t1-w5"]["sse"]["p99"] < 150.0,
     "capacity-five graph-on SSE p99 exceeded 150 ms")

need((r["ordinary-t1-w5"]["draft"], r["ordinary-t1-w5"]["accepted"], r["ordinary-t1-w5"]["hash"]) ==
     (316, 175, "131aac00c3a52f58e4d446e10208f9f271d44afca35abd8b2cae696d7a7789ee"),
     "capacity-five no-hit path changed neural depth/output")
need(max(map(int, r["ordinary-t1-w5"]["proposed"])) <= 4,
     "capacity-five no-hit path widened neural MTP")
need((r["repeat-cap2-w5"]["draft"], r["repeat-cap2-w5"]["accepted"], r["repeat-cap2-w5"]["hash"]) ==
     (44, 40, "06ce020b287f988081bb3e50fe8d4c2754b8a2692fc642bdd2f5f8e3b36a35c1"),
     "explicit cap-two parity changed")
need(max(map(int, r["repeat-cap2-w5"]["proposed"])) <= 2,
     "explicit request cap two lost precedence")

summary = {
    "head": (out / "provenance.txt").read_text().splitlines()[0].split("=", 1)[1],
    "standard": {"temperature_zero": r["standard-t0"], "stochastic": r["standard-t1"]},
    "k4v_temperature_zero": {"width4": r["repeat-t0-w4"], "width5": r["repeat-t0-w5"]},
    "k4v_stochastic": {
        "width4": r["repeat-t1-w4"], "width5": r["repeat-t1-w5"],
        "delta_pct": 100.0 * (r["repeat-t1-w5"]["tps"] / r["repeat-t1-w4"]["tps"] - 1.0),
    },
    "capacity5_no_hit": r["ordinary-t1-w5"],
    "explicit_cap2": r["repeat-cap2-w5"],
}
(out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
print("FINAL_MTP_SINGLE_STREAM_PASS")
PY

PROD_AFTER=$(sha256sum "$PRODUCTION" | awk '{print $1}')
echo "production_after=$PROD_AFTER" | tee -a "$OUT/provenance.txt"
[[ $PROD_BEFORE == "$PROD_AFTER" ]]
sha256sum "$OUT/provenance.txt" "$OUT/diff-stat.txt" "$OUT/version.txt" \
    "$OUT/ctest.log" "$OUT/summary.json" >"$OUT/SHA256SUMS"
echo "final_artifact=$OUT"
