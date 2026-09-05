# speculative sidecars for Qwen3.8

This tree carries optional host-mediated sidecars for Qwen3.8-27B MTP/DFlash2
and Qwen3.8 Flash Next MTP. The target model remains authoritative: sidecars
propose token IDs and the normal target verifier accepts or rejects them.

The sidecars are intentionally **not** enabled by default. Runtime activation
requires the exact opt-in `SPEC_SIDECAR=1`; an unset value, `0`, or any other
value leaves the sidecar code dormant and preserves native speculative
selection. They are model-specific, support up to eight isolated sequences,
and support both keyed stochastic and greedy text inference. The host treats
them as stateful speculative implementations rather than stateless token
generators.

## Automatic provider assets

On Linux, normal sidecar launches prepare provider assets automatically. Users do not need Python, a manual conversion command, provider path variables, or a separately prepared bundle.

Dense MTP needs only its target GGUF:

```sh
SPEC_SIDECAR=1 ./build/bin/llama-server \
  -m /models/Qwen3.8-27B-Q4_0.gguf [...]
```

DFlash and Qwen4Exp use the normal `-md` draft-model argument. When no explicit speculative type is supplied, the draft GGUF metadata selects the matching sidecar type automatically:

```sh
SPEC_SIDECAR=1 ./build/bin/llama-server \
  -m /models/Qwen3.8-27B-Q4_0.gguf \
  -md /models/Qwen3.8-27B-DFlash2-Q4_0.gguf [...]
```

The first launch validates the target and draft, converts only the required tensors, writes a model-keyed bundle under the normal llama.cpp cache, validates the exact provider schema, and atomically commits it. Later launches reuse that bundle. Incomplete, truncated, or schema-invalid automatic entries are rebuilt under a process lock. Source GGUFs are never modified. `--spec-sidecar-cache DIR` or `LLAMA_ARG_SPEC_SIDECAR_CACHE` changes the cache root but is not required.

The dense Qwen3.8 and Qwen3.6 MoE MTP profiles convert compatible target tensors to their fixed provider schemas. Qwen4Exp converts its `-md` GGUF and creates a full identity vocabulary map. DFlash accepts the exact 81-tensor model geometry from canonical Q4_K_M as well as Q4_0, Q8_0, F16, and BF16 sources, then writes the fixed F32/Q4_K/Q6_K runtime schema. DFlash does not use INT5; that format was an unretained experiment.

The bundled draft-vocabulary map is integrity checked. Optional explicit ID and provider paths remain expert overrides, not normal setup requirements. Automatic preparation fails closed with a specific warning for unsupported architectures, shapes, tensor encodings, missing `-md`, unwritable caches, or missing provider libraries. Human-readable GGUF model labels are not required.

### Manual/offline preparation (optional)

The Python tools remain available for inspecting or preparing an explicit offline bundle. They require `gguf-py` from this checkout and NumPy:

```sh
python3 -m pip install -e ./gguf-py
python3 tools/spec-sidecar/prepare_assets.py dflash \
  --target /models/Qwen3.8-27B-Q4_0.gguf \
  --draft /models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --output /artifacts/spec-sidecar-dflash
python3 tools/spec-sidecar/validate_assets.py dflash /artifacts/spec-sidecar-dflash
```

Do not mix an ID table, head, manifest, or weights from different preparation runs.

## Qwen3.6 35B-A3B MoE MTP

`qwen35moe-mtp` is a separate compatibility provider for the Qwen3.6/Qwen3.5 MoE model identified by GGUF as `qwen35moe` (`35B-A3B`). It cannot reuse the dense Qwen3.8-27B provider: the MoE target has a 2,048-wide hidden state, 40 trunk blocks plus one MTP block, 16/2 attention heads, and an 8-of-256 expert MTP FFN.

With exact `SPEC_SIDECAR=1`, a matching target automatically selects the bundled provider, converts the trained MTP block and output head to the Q4_0/F32 cache schema, and uses the built-in integrity-checked 40,960-row ID map. The following Python command is only for an explicit offline bundle:

```sh
python3 tools/spec-sidecar/prepare_assets.py qwen35moe-mtp \
  --target /absolute/models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf \
  --ids /absolute/artifacts/draft_vocab_ids.bin \
  --output /absolute/artifacts/spec-sidecar-qwen35moe-mtp

python3 tools/spec-sidecar/validate_assets.py qwen35moe-mtp \
  /absolute/artifacts/spec-sidecar-qwen35moe-mtp
```

The `LLAMA_SPEC_QWEN35MOE_*` paths remain optional overrides for explicit bundles.

On gfx1030 the provider uses rocPRIM for the 40,960-row stochastic top-k,
cooperatively sorts/remaps its 32 candidates, and routes the 8-of-256 MoE in a
256-lane kernel. The portable scalar paths remain available through
`LLAMA_SPEC_QWEN35MOE_ROCPRIM_TOPK=0` and
`LLAMA_SPEC_QWEN35MOE_PARALLEL_ROUTER=0`. Batched prompt catch-up can be
reverted with `LLAMA_SPEC_QWEN35MOE_BATCHED_CATCHUP=0`.

Automatic activation still requires workload qualification because profitability depends on the request shape. On the validated four-gfx1030 layer-split model, 256-token
code/prose/JSON runs improved 28.6–40.3% over native MTP. A 6,336-token prompt
with only 128 output tokens remained slower end to end because reconstructing
the sidecar's MTP KV is less efficient than the native large-batch graph.
Qualify acceptance and prompt/output balance on the exact deployment workload;
do not reuse the Qwen3.8-27B tensor-split launch policy for this MoE model.

## Flash Next MTP

Flash Next uses its own provider and cannot reuse the 27B bundle. Supply the matching base target and Qwen4Exp MTP GGUF with `-md`; automatic preparation converts its 34 tensors and writes the required full 248,320-row identity map:

```sh
SPEC_SIDECAR=1 ./build/bin/llama-server \
  -m /models/Qwen3.8-Flash-Next-00001-of-00004.gguf \
  -md /models/Qwen3.8-Flash-Next-MTP-Q4_0.gguf [...]
```

The profile requires `qwen4exp`, width 2,560 with a 10,240-wide target handoff, 48 target blocks, 512 experts, and vocabulary 248,320.

## Build

The unified RDNA build script (`scripts/build-rdna-unified.sh`) compiles the
MTP and DFlash sidecars by default (`BUILD_SIDECARS=ON`), passes the selected
architecture, and keeps the unified RCCL build enabled. The RDNA2 build
scripts in this repository (`scripts/build-rdna2-rocm.sh` and
`scripts/build-rdna2-portable.sh`) also compile the sidecar libraries by
default; pass `BUILD_SPEC_SIDECARS=OFF` (or `--no-spec-sidecars` for the
portable script) to skip them. The libraries are dormant unless
`SPEC_SIDECAR=1` is set at runtime.

A plain CMake build does not compile the sidecars. Enable them explicitly in a
HIP build and select the actual GPU architecture:

```sh
cmake -S . -B build-spec-sidecar \
  -G Ninja \
  -DGGML_HIP=ON \
  -DLLAMA_BUILD_SPEC_SIDECARS=ON \
  -DLLAMA_SPEC_SIDECAR_HIP_ARCHITECTURES=gfx1030 \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc
cmake --build build-spec-sidecar --target llama-server
```

When `LLAMA_BUILD_SPEC_SIDECARS=ON`, `llama-server` depends on the complete
registered provider set. A targeted server build therefore produces
`spec_hip_sidecar.so`, `spec_dflash_sidecar.so`,
`spec_qwen35moe_mtp_sidecar.so`, and `spec_qwen4exp_mtp_sidecar.so` in the
build `bin` directory; providers do not need to be named manually. At runtime, the server discovers the matching library beside the executable or in the installed `lib` directory and creates model-derived assets in its cache. Prebuilt directories under `bin/spec-sidecar-*` or installed `share/llama.cpp/spec-sidecar` paths remain optional overrides. These targets are optional because each provider contains fixed model dimensions and is not a replacement for the normal HIP backend.

## Run MTP

The default bundle layout needs only the master gate; provider paths are
optional overrides. For tensor-parallel Qwen3.8 inference where target sampling remains on the CPU, `GGML_TP_SHARDED_OUTPUT=1` enables vocabulary-axis primary output and removes the primary full-logit output AllReduce. The sidecar's compact draft head and provider-local device top-k remain active. If `--backend-sampling` is enabled at model load, the loader safely retains hidden-axis/full-logit output instead so target backend sampling works; vocabulary-axis sharding and target backend sampling are not combined yet. A request-level backend-sampling override cannot change a model already loaded in vocabulary-sharded mode and falls back to CPU there. Leave the variable unset or use `auto` when full logits are required for backend sampling.

```sh
export SPEC_SIDECAR=1
# Optional TP2/TP4 CPU-target-sampling optimization (also enables output sharding):
export GGML_TP_SHARDED_OUTPUT=1

./build-spec-sidecar/bin/llama-server \
  -m /absolute/models/Qwen3.8-27B-Q4_0.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --spec-draft-p-min 0 \
  -np 1 --no-context-shift \
  --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots
```

## Run Flash Next MTP

Supply the matching Qwen4Exp MTP model with `-md`; the provider, assets, and full identity ID map are discovered or prepared automatically.

```sh
SPEC_SIDECAR=1 ./build/bin/llama-server \
  -m /models/Qwen3.8-Flash-Next-00001-of-00004.gguf \
  -md /models/Qwen3.8-Flash-Next-MTP-Q4_0.gguf \
  --spec-type draft-mtp,ngram-map-k4v --spec-draft-n-max 3 \
  --batch-size 128 --ubatch-size 128 --spec-draft-ubatch-size 128 \
  --parallel 1 --ctx-checkpoints 0
```

Batch and ubatch 128 are the validated gfx1030 Flash Attention settings. The
current Flash Next provider supports up to 131,072 positions; larger target
contexts safely fall back after that boundary.

## Run DFlash2

Pass the DFlash GGUF with `-md`. Its metadata selects DFlash and the server automatically prepares the provider cache. After the sidecar probe succeeds, the host does not load a native DFlash model or draft context:

```sh
export SPEC_SIDECAR=1

./build-spec-sidecar/bin/llama-server \
  -m /absolute/models/Qwen3.8-27B-Q4_0.gguf \
  -md /absolute/models/Qwen3.8-27B-DFlash2-Q4_0.gguf \
  --spec-draft-n-max 7 --spec-draft-p-min 0 \
  -np 1 --no-context-shift \
  --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots
```

On gfx1030, long linear catch-up omits K/V projections older than DFlash2's
trained 2,048-token attention window while preserving the original eight-row
kernel boundaries. The logical cursor remains contiguous and skipped rows can
never be read by a later draft. Set
`LLAMA_SPEC_HIP_DFLASH_WINDOW_CATCHUP=0` to restore full-prefix processing.

## State, safety, and current limits

- Up to eight sidecar sequences are supported. Each sequence has an isolated
  logical cursor and KV namespace; KV storage is allocated lazily per active
  sequence. Larger `-np` values use the native drafter or target-only fallback.
- Greedy drafting uses `temperature=0` and `p_min=0`. For `temperature>0`,
  both sidecars sample from a compact top-k q distribution and return that q to
  the target residual verifier. The proposal RNG is a deterministic keyed
  stream derived from the request seed, sequence, position, sidecar kind, and
  draft step; target acceptance/rejection RNG remains owned by the main
  sampler. `p_min` is applied to the sampled q probability. The Qwen3.8 HIP
  MTP provider uses capability-gated rocPRIM top-k when available and retains
  the portable reduction fallback; this is compiled independently for gfx1030
  and gfx1100. On gfx1030, `LLAMA_SPEC_HIP_MTP_PARALLEL_TOPK_FINAL=0` restores
  the scalar final merge for diagnostics. The gfx1030 Qwen35/MTP provider
  likewise uses rocPRIM device top-k when its headers are available and
  otherwise retains the portable two-stage device reduction. These
  provider-local optimizations are
  independent of the Qwen4Exp sidecar.
- Text-only, contiguous positions are the supported sidecar input. Vision
  batches, unsupported interleaving, and migration disable the sidecar safely.
  With a single HIP target ubatch on the matching device, the host passes
  borrowed target device pointers and attaches the sidecar to the target HIP
  stream; the target context defers those host output copies until a host
  getter is requested. Tensor-parallel `Meta()` outputs are exposed only when
  their split state is mirrored; split or partial tensors fail closed to host
  materialization. Compute-arena shard lookup remains tied to the active Meta
  graph plan so a cached descriptor cannot outlive its graph storage. A
  borrowed graph tensor cannot survive a later ubatch, so multi-ubatch logical
  decodes materialize each ubatch in stream order before its graph storage is
  reused. Otherwise the sidecar uses the synchronized host-copy path.
  `LLAMA_SPEC_HIP_DFLASH_DEVICE_INPUT=0` is a diagnostic rollback to force the
  synchronized DFlash host-input path; it does not alter DFlash's fixed draft
  width.
- The ABI exposes sequence-scoped `state_size`, `get_state`, `set_state`,
  `reset_state`, `truncate_state`, `commit_state`, and `rebase_state` for both
  sidecars. Snapshots contain only a position cursor plus an epoch; the large
  device KV cache is not serialized or copied. Target-derived rows are staged
  in pending KV and only the accepted prefix is copied into persistent KV.
  The speculative manager wraps state by implementation type, so stacked
  implementations cannot consume one another's state. DFlash committed KV
  storage starts at 16K rows and grows geometrically up to the effective target
  context; growth recaptures the per-sequence graph once, and
  reset releases grown storage.
- Qwen3.8-27B sidecar KV capacity follows the effective per-sequence target
  context and is allocated on demand. `LLAMA_SPEC_HIP_MAX_POS=N` may impose a lower position
  cap to reduce VRAM use; it cannot raise capacity above the target context and
  does not change the fixed F16 sidecar KV datatype.
- Prompt and ordinary target rows are implicitly committed; target
  verification stages rows and acceptance commits only the accepted prefix.
  Checkpoint rollback discards pending rows and restores the cursor, slot reset
  starts a new epoch, and context shifting rebases committed device KV rows.
  Any failed update or restore enters target-only mode instead of guessing at
  state.
- N-gram and other speculative implementations may remain stacked with MTP
  or DFlash. A sidecar stages/commits target rows even when another
  implementation wins, so it can take over on a later round. The gfx1030
  anti-stutter policy normally caps K4V at the configured neural width. Under
  exact `SPEC_SIDECAR=1`, the qualified dense-MTP+K4V profile automatically
  enables content-aware stacked verification: only the K4V cap may rise through
  MTP width plus the selected boost, while MTP generation remains fixed. Set
  `SPEC_CONTENT_BOOST=0` to retain the prior fixed cap. A
  K4V miss therefore incurs no classifier pass, wider neural draft, or wider
  target pass. Only a K4V candidate longer than the MTP baseline invokes the
  classifier; an emitted longer hit still receives all `N + 1`
  target-verification rows. DFlash retains the baseline selected by
  `--spec-draft-n-max` (not a hard-coded width four) because adding a content
  boost in a matched K4V5 screen was slower even with complete fifth-position
  acceptance.
- Prompt-cache and external slot-file restore do not persist the sidecar's
  device KV contents. If a restored target state does not receive a complete
  contiguous sidecar prefill, the sidecar rejects the gap and the host uses
  target-only mode for correctness.
- The target verifier remains the correctness authority. A sidecar ABI/artifact
  probe runs before draft construction. A successful DFlash probe permits the
  target tensor-parallel output head to remain sharded; if the promised probe
  later fails, host-draft fallback is refused rather than risking a sharded-head
  mismatch. A successful probe selects sidecar-only mode and avoids loading the
  host draft model/context; a later HIP initialization/runtime failure disables
  drafting and enters target-only mode rather than loading a late or potentially
  desynchronized native cache.
- Validate the activation log and artifact set before making performance
  comparisons. speculative sidecar's published numbers are Windows/RX7900 XTX
  research evidence, not RDNA2/Linux qualification.
