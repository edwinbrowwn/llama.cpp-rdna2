# BridgeSpec Qwen3.8-27B sidecars

This tree carries an optional, host-mediated BridgeSpec integration for the
Qwen3.8-27B MTP and DFlash2 drafters. The target model remains authoritative:
the sidecar proposes token IDs and the normal target verifier accepts or
rejects them.

The sidecars are intentionally **not** enabled by default. They are
model-specific, currently single-instance, and initially support greedy text
inference only.

## Prepare the 27B artifacts

The preparation tools require `gguf-py` from this checkout and NumPy:

```sh
python3 -m pip install -e ./gguf-py
```

Obtain the 40,960-ID draft vocabulary from the matching, licensed source and
keep it outside the source tree. Then prepare the artifacts from a compatible
Qwen3.8-27B Q4_0 target and the matching DFlash2 Q4_K_M draft:

```sh
python3 tools/bridgespec/prepare_assets.py mtp \
  --target /absolute/models/Qwen3.8-27B-Q4_0.gguf \
  --ids /absolute/artifacts/draft_vocab_ids.json \
  --output /absolute/artifacts/bridgespec-mtp

python3 tools/bridgespec/prepare_assets.py dflash \
  --target /absolute/models/Qwen3.8-27B-Q4_0.gguf \
  --draft /absolute/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --ids /absolute/artifacts/draft_vocab_ids.json \
  --output /absolute/artifacts/bridgespec-dflash

python3 tools/bridgespec/validate_assets.py mtp /absolute/artifacts/bridgespec-mtp
python3 tools/bridgespec/validate_assets.py dflash /absolute/artifacts/bridgespec-dflash
```

The target must have Q4_0 token embeddings, a Q6_K output head, vocabulary
size 248,320, and one MTP block at index 64. The DFlash draft must provide the
81-tensor Qwen3.8-27B DFlash2 schema. Do not mix an ID table, sliced head, and
weights from different preparation runs. The generated MTP `*-bridgespec.gguf`
is retained as a prepared derivative, but the initial sidecar host path uses
the original full-vocabulary target because sliced native MTP-head loading is
not enabled in this integration yet.

## Build

Normal builds do not compile the sidecars. Enable them explicitly in a HIP
build and select the actual GPU architecture:

```sh
cmake -S . -B build-bridgespec \
  -G Ninja \
  -DGGML_HIP=ON \
  -DBRIDGESPEC_BUILD_SIDECARS=ON \
  -DBRIDGESPEC_HIP_ARCHITECTURES=gfx1030 \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc
cmake --build build-bridgespec \
  --target bridgespec-hip-mtp bridgespec-hip-dflash llama-server
```

The resulting libraries are `spec_hip_sidecar.so` and
`spec_dflash_sidecar.so` in the build `bin` directory. The sidecar target is
optional because it contains fixed Qwen3.8-27B dimensions and is not a
replacement for the normal HIP backend.

## Run MTP

Use an absolute sidecar library and artifact path:

```sh
export LLAMA_SPEC_HIP_SIDECAR=/absolute/build/bin/spec_hip_sidecar.so
export LLAMA_SPEC_HIP_WEIGHTS=/absolute/artifacts/bridgespec-mtp
export LLAMA_DRAFT_HEAD_IDS=/absolute/artifacts/bridgespec-mtp/draft_head_ids.bin

./build-bridgespec/bin/llama-server \
  -m /absolute/models/Qwen3.8-27B-Q4_0.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --spec-draft-p-min 0 \
  -np 1 --no-context-shift \
  --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots
```

## Run DFlash2

DFlash still uses its matching GGUF for metadata and native fallback; keep its
weights on the host with `-ngld 0`. The sidecar loads the prepared controller
artifacts:

```sh
export LLAMA_SPEC_HIP_DFLASH=/absolute/build/bin/spec_dflash_sidecar.so
export LLAMA_SPEC_HIP_DFLASH_DIR=/absolute/artifacts/bridgespec-dflash

./build-bridgespec/bin/llama-server \
  -m /absolute/models/Qwen3.8-27B-Q4_0.gguf \
  -md /absolute/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  -ngld 0 --spec-type draft-dflash \
  --spec-draft-n-max 7 --spec-draft-p-min 0 \
  -np 1 --no-context-shift \
  --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots
```

## Safety and current limits

- `-np 1` is required; the sidecar ABI has one process-global state object.
- Greedy drafting is required (`temperature=0`, `p_min=0`). The host disables
  a loaded sidecar rather than silently using greedy proposals for stochastic
  requests.
- Text-only, contiguous positions are the initial supported path. Vision,
  sequence forks, save/restore, prompt-cache reuse, context shifting, and
  migration are not supported.
- The target verifier remains the correctness authority. An initialization
  failure falls back to the native drafter. A runtime sidecar failure disables
  drafting for the process rather than using a potentially desynchronized
  native cache.
- Validate the activation log and artifact set before making performance
  comparisons. BridgeSpec's published numbers are Windows/RX7900 XTX
  research evidence, not RDNA2/Linux qualification.
