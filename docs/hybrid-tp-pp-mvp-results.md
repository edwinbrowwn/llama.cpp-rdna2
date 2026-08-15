# Hybrid tensor-parallel × pipeline-parallel MVP results

## Status and decision

Branch: `feat/hybrid-tp-pp-mvp`

Base: `db3eb285898d83883203bc2284307f4f7f544051`

Measured implementation: explicit tensor mode with TP=2 inside each PP stage and PP=2 across four selected devices:

- stage 0: `Meta(ROCm0,ROCm1)`, transformer layers 0 through 31;
- stage 1: `Meta(ROCm2,ROCm3)`, transformer layers 32 through 63 plus auxiliary/NextN/output work.

The implementation is technically successful: it preserves the legacy PP1 path, passes the scoped correctness/regression gates, uses direct asynchronous rank-wise Meta transport with no measured PP fallback, and has a rocprof device timeline proving Stage0(B) overlaps Stage1(A) at concurrency 4.

**Recommendation:** stop productionization/default work. Retain this topology as an explicit experimental opt-in only. It improves service capacity over two-GPU TP2 at higher concurrency and longer prompts, but every matched same-four-GPU comparison favors legacy TP4. Do not merge it to `master` without a separate explicit review and request.

## Required invocation and gates

The measured hybrid path requires:

```text
-ngl all
-dev ROCm0,ROCm1,ROCm2,ROCm3
--split-mode tensor
-ts 1,1
--tp-size 2
--pp-size 2
--pp-split 1,1
--fit off
-fa on
```

The benchmark environment is:

```text
HSA_OVERRIDE_GFX_VERSION=10.3.0
HSA_NO_SCRATCH_RECLAIM=1
GGML_CUDA_DISABLE_GRAPHS=1
GGML_CUDA_ALLREDUCE=nccl
GGML_META_COPY_DEBUG=1
GGML_SCHED_TRACE=0
GGML_TP_SHARDED_OUTPUT=0
LD_LIBRARY_PATH=<worktree>/build/bin:/opt/rocm/core-7.14/lib
ulimit -s 8192
```

Performance runs fail closed unless every active hybrid Meta summary has:

- `success == meta_attempts`;
- `meta_fallback == 0`;
- `unsupported_state == 0`;
- physical rank-wise bytes equal to twice the logical bytes.

Expected CPU/user-input fallback is counted separately and is not treated as PP fallback.

## Correctness and regression evidence

Qualified gates include:

- pure CPU (`-ngl 0 -dev none`), single GPU, legacy layer split, TP2, and TP4;
- topology, group-local split, owning output/NextN heads, Meta events, and Meta async-copy tests;
- two independent two-rank hybrid communicators and no normal four-rank hybrid collective;
- real Qwen3.8 multi-sequence state removal/restore and host/on-device migration;
- synthetic recurrent rollback and replay;
- same-model and external-Q4 MTP accept/reject rollback through concurrent context growth;
- Qwen3.6 optimized output-head on/off behavior;
- real Qwen3.8 full-logit capture.

Hybrid and TP2 logits were byte-identical for 993,280 F32 values in the deterministic capture. TP4 used a different reduction order but retained identical argmax at all positions, top-10 overlap of at least 9/10, top-50 overlap of at least 48/50, correlation of at least 0.9998499362, maximum absolute difference 0.1270676, and maximum RMSE 0.0256296.

The full-size Qwen3.8 restore-over-dirty-context mismatch reproduces identically on legacy TP2. Long-context fixed-seed no-spec/MTP output divergence also reproduces on legacy TP2. Both are retained as inherited behavior rather than attributed to pipeline placement.

## Device overlap and synchronization

The scheduler uses the existing four copy slots and composite events. No input wait, event wait, graph-reuse synchronization, or ownership rule was removed.

The retained rocprof evidence shows:

- CLI concurrency 1: 9 adjacent stage pairs, zero Stage1(A)/Stage0(B) kernel-busy overlap;
- server concurrency 4: 41 adjacent pairs, 10 with positive overlap, 730,556,554 ns aggregate overlap, and 120,626,917 ns maximum overlap;
- 42/42 PP handoffs were asynchronous with no synchronous PP fallback;
- all four scheduler slots were used;
- 21 graph-reuse full synchronization calls remained in place.

This proves real device execution overlap, not merely host enqueue overlap.

## Performance results

Target model: `Qwen3.8-27B-Q8_0.gguf`.

External draft: `Qwen3.8-27B-MTP-Draft-Q4_0.gguf`, `draft-mtp`, `n_max=2`, draft devices `ROCm2,ROCm3`.

All table values are aggregate generated tokens per second. Each point has three repeats.

### PP512 / TG128

| configuration | C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|
| TP2 PP1 no-spec | 23.727 | 44.507 | 62.702 | 70.250 |
| TP4 PP1 no-spec | 27.636 | 52.176 | 78.890 | 95.558 |
| TP4 PP1 external MTP | 49.513 | 80.393 | 87.704 | 103.164 |
| TP2×PP2 no-spec | 23.584 | 44.444 | 66.522 | 78.126 |
| TP2×PP2 external MTP | 45.013 | 70.871 | 73.267 | 93.271 |

Hybrid no-spec beats TP2 by 6.09% at C4 and 11.21% at C8, but trails TP4 by 15.68% and 18.24%. This is a capacity gain from using twice as many GPUs, not a same-hardware efficiency gain.

### Long prefill, C4 / TG128

| PP | TP2 no-spec | TP4 no-spec | TP4 MTP | hybrid no-spec | hybrid MTP |
|---:|---:|---:|---:|---:|---:|
| 2048 | 35.041 | 48.956 | 50.638 | 42.852 | 45.472 |
| 4096 | 21.230 | 32.580 | 33.159 | 29.069 | 31.059 |

At PP2048/4096, hybrid no-spec beats TP2 by 22.29%/36.93%, but remains 12.47%/10.78% behind TP4. The same-hardware gap narrows but does not reverse.

### PP512 / TG512

| configuration | C1 | C4 |
|---|---:|---:|
| TP2 PP1 no-spec | 23.613 | 71.454 |
| TP4 PP1 no-spec | 27.708 | 89.068 |
| TP4 PP1 external MTP | 49.788 | 114.881 |
| TP2×PP2 no-spec | 23.473 | 74.618 |
| TP2×PP2 external MTP | 46.341 | 96.770 |

At C4, hybrid no-spec is 4.43% above TP2 but 16.22% below TP4. Hybrid external MTP improves over hybrid no-spec by 29.69%, yet remains 15.77% behind TP4 external MTP.

### PP boundary sweep

Measured complete-layer assignments were 28/36, 32/32, 36/28, and 40/24 at C1/C2/C4 for no-spec and external MTP. At C4, alternatives changed no-spec throughput relative to 32/32 by -1.46%, -0.98%, and -3.82%; MTP changed by -9.50%, -0.51%, and -2.61%. Equal 32/32 is the best consistent result.

## MTP interpretation and seed policy

The external Q4 draft is beneficial inside hybrid, especially for longer generation. Same-model native MTP was much slower in the retained concurrency-2 diagnostic and is not recommended.

The benchmark harness now supplies fixed per-request seeds (`20260815 + request index`) and writes `output-equivalence.json`. Strict hybrid equality can be enabled with `REQUIRE_MTP_OUTPUT_EQUIVALENCE=1` for bounded correctness workloads. It is audit-only for broad performance workloads because long-context recurrent/speculative divergence reproduces on legacy TP2.

Earlier performance matrices predate explicit seed plumbing. Their prompts, generated-token counts, topology, and service timing remain exact; their MTP acceptance/output paths are service samples rather than deterministic equivalence evidence. The fixed-seed TG512 matrix and paired diagnostics make this limitation explicit.

Target-versus-draft extraction/readback time is not separately exposed by the backend. Available cumulative draft-generation and acceptance-bookkeeping timing is preserved without inventing unavailable target timing.

## Reproduction

Run one matched matrix from a clean worktree with a fresh output directory:

```bash
cd /home/edwin/llama.cpp-rdna2-hybrid-tp-pp
OUT=/home/edwin/hybrid-tp-pp-evidence/performance-mvp/recheck \
  PP_SPLIT=1,1 SEED_BASE=20260815 REQUIRE_MTP_OUTPUT_EQUIVALENCE=0 \
  CONCURRENCY=1,4 PARALLEL=4 PROMPT_TOKENS=512 N_PREDICT=512 \
  REPEATS=3 CTX_PER_SEQ=2048 BATCH=1024 UBATCH=512 BASE_PORT=18600 \
  ./scripts/run-hybrid-performance-matrix.sh
```

The runner refuses to overwrite an existing output directory, verifies communicator/pipeline shape and truncation, audits fixed-seed outputs, enforces zero hybrid Meta fallback, emits JSON/CSV/Markdown summaries, samples GPU activity/VRAM/temperature, and writes SHA-256 manifests.

Primary preserved evidence is under `/home/edwin/hybrid-tp-pp-evidence/`, especially:

- `device-timeline/`;
- `state-mutation/` and `mtp-state-stress/`;
- `numerical-equivalence/` and `legacy-regressions/`;
- `performance-mvp/pp512-tg128-c1c2-r3-68fb791c6/`;
- `performance-mvp/pp512-tg128-c4c8-r3-4950a1605/`;
- `performance-mvp/boundary-sweep-pp512-tg128-c1c2c4-r3-4950a1605/`;
- `performance-mvp/pp2048-tg128-c4-r3-4950a1605/`;
- `performance-mvp/pp4096-tg128-c4-r3-4950a1605/`;
- `performance-mvp/pp512-tg512-c1c4-r3-447dc0ab7/`.

The final external verifier and complete evidence index are recorded separately in the task completion evidence.
