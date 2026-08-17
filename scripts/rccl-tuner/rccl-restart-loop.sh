#!/usr/bin/env bash
set -euo pipefail
W=${W:-/home/edwin/llama.cpp-rdna2-rccl-autotune}
D=$W/scripts/rccl-tuner
ROOT=${ROOT:-/home/edwin/models/qwen38-27b-q4s8}
M=${M:-$ROOT/autoround-q4-fast/Qwen3.8-27B-Q4_0.gguf}
E=${E:-$ROOT/rccl-autotune-20260817/phase8/restarts}
PORT=${PORT:-18100}
ROUNDS=${ROUNDS:-20}
mkdir -p "$E"
export HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_HIP_SAFE_STATE_IO=1 GGML_HIP_GFX1030_Q8_CACHE=1 GGML_HIP_GFX1030_GDN_SIBLING_FUSION=1 GGML_HIP_GFX1030_Q8_1_FUSION=1 GGML_HIP_GFX1030_NATIVE=1 GGML_HIP_GFX1030_ADD_RMS_NORM_FUSION=1 NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=PXB GGML_TP_SHARDED_OUTPUT=1 GGML_CUDA_ALLREDUCE=nccl HSA_OVERRIDE_GFX_VERSION=10.3.0 HSA_NO_SCRATCH_RECLAIM=1 GGML_CUDA_P2P=1 GGML_HIP_GRAPHS=1 GGML_HIP_RCCL_TUNE=force
unset NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS NCCL_NTHREADS NCCL_DEBUG NCCL_DEBUG_SUBSYS NCCL_TUNER_CONFIG_FILE
for i in $(seq 1 "$ROUNDS"); do
  log="$E/server-$i.log"
  start=$(date +%s%N)
  "$D/llama-rdna2-rccl-auto.sh" -m "$M" --host 127.0.0.1 --port "$PORT" --no-webui --ctx-size 2048 --parallel 1 --flash-attn on -ngl 999 -sm tensor -dev ROCm0,ROCm1,ROCm2,ROCm3 -ts 1,1,1,1 >"$log" 2>&1 &
  pid=$!
  ready=0
  for j in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:$PORT/health" >"$E/health-$i.json" 2>/dev/null; then ready=1; break; fi
    if ! kill -0 "$pid" 2>/dev/null; then break; fi
    sleep 1
  done
  if [ "$ready" != 1 ]; then
    echo "restart $i failed to become healthy" >&2; tail -100 "$log" >&2
    kill -TERM "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; exit 1
  fi
  printf '%s\n' '{"prompt":"Return the word ready.","n_predict":4,"temperature":0,"cache_prompt":false}' | curl -fsS "http://127.0.0.1:$PORT/completion" -H content-type:application/json -d @- >"$E/completion-$i.json"
  python3 -c "import json; x=json.load(open('$E/completion-$i.json')); assert x.get('content') is not None"
  kill -TERM "$pid" 2>/dev/null || true
  for j in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  if kill -0 "$pid" 2>/dev/null; then kill -KILL "$pid" 2>/dev/null || true; fi
  wait "$pid" 2>/dev/null || true
  end=$(date +%s%N)
  printf '%s\t%s\t%s\n' "$i" "$ready" "$(( (end-start)/1000000 ))" >>"$E/runs.tsv"
  if pgrep -f "llama-server.*--port $PORT" >/dev/null; then echo "stale server after restart $i" >&2; exit 1; fi
done
printf 'rounds=%s\nhealthy=%s\n' "$ROUNDS" "$(wc -l < "$E/runs.tsv")" >"$E/summary.env"
sha256sum "$E"/* >"$E/SHA256SUMS"
echo RCCL_RESTART_LOOP=PASS
