# Hybrid TP×PP base audit and frozen baselines

## Frozen revision

- UTC capture: `2026-08-15T00:00:57Z`
- Branch: `feat/hybrid-tp-pp-mvp`
- Worktree: `/home/edwin/llama.cpp-rdna2-hybrid-tp-pp`
- Exact base: `db3eb285898d83883203bc2284307f4f7f544051`
- Base subject: `db3eb2858 docs: add Laguna tensor-split verifier`
- Fork: `https://github.com/edwinbrowwn/llama.cpp-rdna2.git`
- Upstream: `https://github.com/ggerganov/llama.cpp.git`
- Worktree was clean and equal to `fork/master` at capture.

Raw immutable evidence is retained under:

```text
/home/edwin/hybrid-tp-pp-evidence/base-db3eb2858/
```

## Build and runtime freeze

```text
kernel: 7.0.12-v620alluclk1test1
ROCM-SMI: 4.0.0+2b22ab01
HSA runtime: 1.21
ROCm clang: 23.0.0git 46fcb339... + patched 440716f...
CMake: 4.2.3
```

Build cache:

```text
AMDGPU_TARGETS=gfx1030
CMAKE_BUILD_TYPE=Release
GGML_HIP=ON
GGML_HIP_GRAPHS=ON
GGML_HIP_NO_VMM=ON
GGML_HIP_RCCL=ON
GGML_NATIVE=ON
LLAMA_BUILD_TESTS=ON
```

Runtime environment for frozen tests:

```text
HSA_OVERRIDE_GFX_VERSION=10.3.0
HSA_NO_SCRATCH_RECLAIM=1
GGML_CUDA_DISABLE_GRAPHS=1
GGML_CUDA_ALLREDUCE=nccl
LD_LIBRARY_PATH=<worktree>/build/bin:/opt/rocm/core-7.14/lib
ulimit -s 8192
```

The preserved build in `<worktree>/build` reports:

```text
version: 10635 (db3eb2858)
```

Models:

```text
target: /home/edwin/models/qwen38-27b-q4s8/unsloth-q8/Qwen3.8-27B-Q8_0.gguf
sha256: a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348
mtp:    /home/edwin/models/qwen38-27b-q4s8/draft-q4/Qwen3.8-27B-MTP-Draft-Q4_0.gguf
sha256: a79d1b93250e68ed38dea083ca4185ddbc488861a9e2e0d51cd68b05a1ba4bb0
```

## Source audit commands

The host did not initially contain `rg`. Ripgrep 15.1.0 was installed non-root at `/home/edwin/.local/bin/rg`, then the requested audit was run exactly from the frozen worktree:

```bash
rg -n 'n_devices\(\)' src common include ggml
rg -n 'get_split_state_ud' .
rg -n 'tensor_split|tensor-split' src common include
rg -n 'LLAMA_SPLIT_MODE_(TENSOR|LAYER|ROW|NONE)' src common include ggml
rg -n 'pipeline_parallel' src common ggml
rg -n 'cpy_tensor_async|event_(new|free|record|wait|synchronize)' src common ggml
rg -n 'GGML_TP_|no_tp_output|load_mtp|ctx_other|speculative' src common
rg -n 'dev_layer\(|dev_output\(' src
```

All raw outputs and SHA-256 checksums are under:

```text
/home/edwin/hybrid-tp-pp-evidence/base-db3eb2858/source-audit/
```

## Semantic classification

Classes:

- **A — logical PP stage:** count/ownership of logical model devices after hybridization.
- **B — physical TP group:** count/split geometry of child GPUs inside one Meta stage.
- **C — legacy-only:** preserve unchanged when PP is disabled.
- **D — gated/deferred:** unsupported for initial hybrid performance path or requires explicit policy.

### `n_devices()` and split-state userdata

| Area | Current meaning | Hybrid classification/action |
|---|---|---|
| `llama_model::n_devices()` | `devices.size()` | **A**: becomes PP stage count only in explicit hybrid mode; unchanged in legacy paths. |
| `llama-context.cpp` scheduler eligibility | number of logical model devices | **A**: topology predicate must replace layer-split enum inference. |
| `llama-model.cpp::load_tensors()` split loops | outer model-device/layer placement | **A**: use normalized PP split in hybrid; existing tensor split in legacy. |
| `llama_meta_device_get_split_state_userdata::n_devices` | child GPUs in the one Meta device | **B**: each group owns stable userdata and its own count. |
| `llama_meta_device_get_split_state()` | gets split from `ud->model->tensor_split()` | **B**: hybrid callback must consume group-owned TP split, never PP split. |
| `llama_prepare_model_devices()` explicit/default TP paths | writes one model-global userdata and creates one Meta | **C/B**: leave PP1 branch byte-for-byte/narrowly unchanged; explicit hybrid branch creates multiple stable groups. |
| output/NextN TP head initialization | one global `get_split_state_ud.n_devices` | **B**: resolve output-owning/layer-owning group. |
| `qwen35moe.cpp` / `deepseek4.cpp` graph config | reads model-global TP count and split | **B**: uniform MVP still needs a group-aware accessor; do not let logical PP stage count leak into optimized TP graph configuration. |
| Meta device identity | child list + callback + raw userdata pointer | **B/lifetime-critical**: userdata must live at a stable address for model lifetime. |

### `tensor_split`

| Area | Classification/action |
|---|---|
| Public `llama_model_params::tensor_split` and common CLI array | **C/B**: exact legacy semantics under PP1; group-local TP geometry under explicit hybrid. Add separate PP configuration rather than reinterpret this field. |
| Model-owned copy (`tensor_split_owned`) | **B**: retain for PP1; topology groups need owned per-group slices. |
| Meta split callback | **B**: use group split pointer/count. |
| `load_tensors()` cumulative split and layer placement | **A** in hybrid, **C** otherwise: select PP split only when hybrid. |
| `make_gpu_buft_list()` | Existing row/layer buffer selection plumbing; **C** unless a verified hybrid issue appears. |
| output/NextN split validation | **B**: owning group split/count. |
| fork MMQ configs (`qwen35moe`, `deepseek4`) | **B**: use the active/owning TP group; existing model-specific support remains separate from generic topology. |
| `common_fit_params()` | **D**: current fitter writes/interprets one global physical-device split. Explicit hybrid must reject fit-on for MVP. |
| DFlash/DSpark draft split override | **D/preserved policy**: external draft remains layer-split; new target hybrid fields must be explicitly reset rather than inherited through `common_params` copy. |

### Split modes

- No new enum is needed. `LLAMA_SPLIT_MODE_TENSOR` continues to mean TP inside each stage.
- Explicit `tp_size/pp_size/pp_split` (or equivalent narrow API) selects hybrid topology.
- `LLAMA_SPLIT_MODE_NONE`, `LAYER`, and `ROW` are **C** and remain on existing paths.
- Architecture support check for tensor mode remains applicable; hybrid must not weaken it.
- DFlash/DSpark external draft conversion from tensor to layer split is **D/preserved**.

### Pipeline scheduler

| Hit | Classification/action |
|---|---|
| Context eligibility: `model.n_devices() > 1` | **A** logical stages. |
| Hard `split_mode == LAYER` gate | Replace only with narrow topology capability allowing hybrid; preserve remaining safety checks. |
| Full GPU offload, KQV offload, no overrides | Retain for measured hybrid MVP. |
| Backend async/events gate | Meta currently fails this; implement real generic capability before claiming performance. |
| Scheduler creation `parallel=cparams.pipeline_parallel` | Reuse existing scheduler first. |
| Graph reuse full scheduler synchronize before `set_inputs()` | Correctness-critical. Instrument count/time; replace only with proven safe slot ownership if it blocks overlap. |

### Copy/events

- Generic copy fallback synchronizes source and destination then performs blocking copy when destination async copy is absent/fails.
- Scheduler split-copy has the same observable fallback pattern and owns up to four copy slots (`GGML_SCHED_MAX_COPIES=4`).
- Meta currently hardcodes `caps.events=false`; `event_new/free/synchronize`, `event_record/wait`, and `cpy_tensor_async` are all null.
- CUDA/HIP child backends already expose events and asynchronous same-device/peer D2D copy, including source event record and destination stream wait.
- **Required generic work:** composite Meta events and all-ranks-prevalidated Meta→Meta async delegation. Do not add HIP code to Meta.
- **Performance validity:** instrument and require zero normal PP sync fallback.

### Device/state ownership

- `dev_layer(il)` drives normal KV allocation, recurrent allocation, DSV4 KV, descriptor placement, and several context operations: **A**, naturally stage-local after correct outer placement.
- `dev_output()` drives output buffers and output operations: **A**, final stage under normal architecture rules.
- These paths still require explicit state mutation, speculative rollback, memory, and placement tests; code shape alone is not proof.

### Speculative decoding

- Ten modes plus none are present exactly as expected.
- `common_base_params_to_speculative()` starts with `common_params result = params`: new hybrid fields would be inherited unless explicitly resolved/reset.
- Same-model MTP creates a second context over the target model, so target topology is naturally shared.
- External draft MTP loads separate model params; its topology policy must be independent and explicit.
- MTP consumes NextN embeddings via host-visible APIs: profile this separately from PP.
- Eagle/DFlash/DSpark can consume intermediate features, so topology APIs cannot assume only final-stage logits matter.
- Initial MVP fully validates MTP; other modes remain architecturally available and preserve existing placement policies.

## Deviations or additional findings versus the supplied review

The supplied design facts match base `db3eb2858`; only navigation line numbers moved. Additional fork-specific audit items that must not be missed:

1. There are two one-Meta construction branches in `llama_prepare_model_devices()` (explicit devices and auto-selected devices); legacy behavior exists in both.
2. Qwen35MoE and DeepSeek4 graph builders read `get_split_state_ud.n_devices` directly, not only the output-head helper. These accesses need an explicit group-size policy before multiple groups can be considered universally correct.
3. `common_params::fit_params` defaults to true; hybrid must fail clearly unless the caller supplied `--fit off`.
4. The current TP4+external-MTP run creates two separate four-rank communicators (target and draft). This is expected legacy behavior and provides a useful regression baseline.
5. Meta device registry construction remains non-thread-safe; serial startup is an explicit MVP constraint.

## Frozen baseline command and workload

Rerunnable script:

```bash
/home/edwin/hybrid-tp-pp-evidence/base-db3eb2858/run-baselines.sh
```

The script uses the frozen build/model hashes and:

```text
context=512
output tokens=128
prompt="Count from one to one hundred and twenty-eight, putting each number on a separate line."
reasoning=off, temp=0, ignore-eos, no warmup
full GPU offload, flash attention on, fit off
split-mode=tensor
tensor split=1,1 for TP2; 1,1,1,1 for TP4
MTP: external Q4_0 draft, n_max=2, all draft layers offloaded
```

Three independent model-load/generation repetitions:

| Case | Prompt tok/s | Generation tok/s |
|---|---:|---:|
| TP2 PP1 no-spec | 80.600 mean (`83.4,79.6,78.8`) | 23.800 mean (`23.8,23.8,23.8`) |
| TP4 PP1 no-spec | 64.933 mean (`66.4,63.4,65.0`) | 27.900 mean (`27.9,27.9,27.9`) |
| TP4 PP1 MTP | 49.067 mean (`49.1,49.2,48.9`) | 59.967 mean (`60.0,60.1,59.8`) |

Verbose MTP diagnostic at matching settings:

```text
84/86 speculative tokens accepted (0.97674)
43/43 draft calls accepted
acceptance by position=(1.000,0.953)
generation=59.6 tok/s
```

This is a concurrency-1 frozen smoke baseline, not the final server throughput matrix. The final MVP must use the same target/draft and matched server concurrency 1/2/4/8 with trace/copy telemetry.
