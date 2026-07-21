# Sequence Fork / Turn-Boundary Snapshot Progress

Last updated: 2026-07-20

## Goal

Replace in-place recurrent checkpoint restore on AMD with a turn-boundary sequence fork/switch lifecycle that:

- preserves prefix reuse for Qwen hybrid/recurrent models;
- keeps restored/branched prompt processing on normal tile flash attention;
- produces the same logits as clean recomputation;
- works first on one GPU, then four-GPU tensor parallelism;
- supports MTP and eventually unified KV plus four server slots;
- avoids host checkpoint serialization on the common path;
- avoids vector-FA prompt fallbacks and full prompt reprocessing.

## Production Baseline

- Production branch: `exp-gpu-sampling`
- Safe baseline commit: `1a3578dd6` (`server: revert AMD checkpoint restore mitigations`)
- Reverted mitigation commits:
  - `e08ba7a10` — preserve AMD recurrent checkpoints with FA fallback
  - `111f35e6b` — reset HIP graphs after checkpoint restore
  - `15271ec21` — keep restored prompts off AMD tile FA
- Production binary rebuilt as server version `10093`.
- AMD context checkpoints are disabled again by the safe baseline.
- Verified retained optimization ancestry:
  - `9bb8bb383` — vocabulary-parallel output sampling
  - `c78889b24` — ROCm/output-mode documentation
  - `4e389bc46` — vocabulary-parallel MTP sampling
  - `cd03d6e5a` — vocabulary-parallel MTP sampling follow-up
  - `fa31a460b` — conservative AMD checkpoint disable

Only the three later checkpoint mitigation commits were reverted; all TP output, RCCL, MTP, grammar fallback, and sampling work remains present.

## Experimental Branch

- Branch: `exp-sequence-fork`
- Worktree: `~/llama-cpp-sequence-fork`
- Starting commit: `1a3578dd6`

## Primary Iteration Model

Qwen3.6 35B-A3B MTP:

```text
~/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
```

Optional multimodal projector for final server integration only:

```text
~/models/Qwen3.6-35B-A3B-GGUF/mmproj-F16.gguf
```

The standalone memory/logit harness will not load the multimodal projector.

## Final Acceptance Model

Qwen3.5 122B-A10B MTP:

```text
~/models/Qwen3.5-122B-A10B-MTP-UD-Q4_K_M/UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf
```

## Existing Primitives to Validate

- `llama_memory_seq_cp()` already copies logical sequence memory.
- `llama_memory_hybrid::seq_cp()` forwards to attention and recurrent memory.
- Recurrent `seq_cp()` shares the source cell with the destination.
- Recurrent `find_slot()` performs copy-on-write when a shared sequence is modified.
- Attention KV sequence copy is expected to share prefix cells.

The first prototype should use these existing operations without server changes.

## Test Pyramid

### Gate 1 — Standalone, 35B, one GPU

- [x] Load model once with a short context and at least two sequence IDs.
- [x] Evaluate deterministic prefix on sequence 0.
- [x] Fork sequence 0 to shadow sequences with `llama_memory_seq_cp()`.
- [x] Continue forked and independently recomputed sequences with the same suffix.
- [x] Compare forks against clean recomputation using a measured cross-sequence variability floor.
- [x] Verify same-sequence clean recomputation is bitwise identical.
- [x] Verify argmax tokens are identical.
- [x] Measure fork and first-write copy-on-write latency.
- [x] Repeat 100 fork/continue/release cycles and verify source-state preservation.

### Gate 2 — Standalone feature ladder

- [x] Tile FA on one GPU.
- [x] HIP graphs (compiled/enabled path used by ROCm build).
- [ ] MTP target/draft state.
- [x] 16k prefix on one GPU.
- [ ] Varied suffix sizes.
- [x] Unified KV.

### Gate 3 — 35B four-GPU TP

- [x] Repeat deterministic clean-vs-fork fixtures.
- [x] Verify TP argmax tokens match one-GPU reference.
- [x] Verify source sequence state remains byte-for-byte unchanged.
- [x] Measure fork overhead and suffix decode throughput.
- [x] Run repeated forks at a 16k prefix.
- [x] Verify vocabulary-sharded output remains compatible.
- [ ] Stress varied suffix sizes at longer context.

### Gate 4 — Targeted 122B confirmation

- [ ] Basic clean-vs-fork correctness.
- [ ] Repeated fork lifecycle.
- [ ] Long-prefix branch test.
- [ ] MTP state alignment.

### Gate 5 — Server integration

Only after Gates 1–4 pass:

- [ ] Internal sequence-ID pool / shadow sequence mapping.
- [ ] Turn-boundary snapshot, continuation fork, commit, and discard.
- [ ] Cancellation and error cleanup.
- [ ] Pi token-transition audit.
- [ ] Four parallel slots.
- [ ] Unified-KV contention.
- [ ] Exact captured Pi sessions.

## Acceptance Criteria

- Forked logits match clean recomputation.
- Forked sampled tokens are identical.
- No tile-FA page faults.
- No host checkpoint load on the common fork path.
- Fork/switch overhead target: less than 500 ms.
- Suffix PP target: within 10–15% of append-only tile PP at the same position.
- Generation throughput unchanged.
- No leaked sequence IDs or unreclaimed KV/recurrent cells.

## PP Optimization Follow-Up

After fork correctness:

1. adaptive physical microbatch size by prefix length;
2. separate MTP hidden-state output from vocabulary logits;
3. F16 vs Q8 KV bandwidth benchmarking;
4. profile long-context RDNA2 tile/stream-K behavior.

These are intentionally deferred until the fork path is correct.

## Current Status

- [x] Reverted the three checkpoint mitigation commits in production source history.
- [x] Rebuilt safe production binary version 10093.
- [x] Pushed safe production branch to `fork/exp-gpu-sampling`.
- [x] Created isolated `exp-sequence-fork` worktree.
- [x] Verified all prior TP output/RCCL/MTP commits remain ancestors of the experimental branch.
- [x] Inspected existing sequence-copy tests/examples.
- [x] Chose a focused `tests/test-sequence-fork.cpp` harness using existing `llama_memory_seq_cp()`.
- [x] Build and run Gate 1 harness.
- [x] Pass deterministic tiny-Qwen CPU control.
- [x] Pass 35B one-GPU 100-cycle test.
- [x] Pass 35B four-GPU TP basic and vocabulary-sharded tests.
- [x] Pass 35B 16k-prefix tests on one GPU and TP.
- [ ] Add deterministic multi-token continuation comparison.
- [ ] Add MTP target/draft fork-state validation.

## Decision Log

- Do not modify `llama-server` until standalone sequence-fork correctness is proven.
- Use 35B for iteration speed; use 122B only for targeted acceptance.
- Start on one GPU without TP, MTP, graphs, or unified KV.
- Add one feature at a time; do not use a combinatorial test matrix.
- Keep production and experimental builds isolated.

## Test Results

### Deterministic tiny-Qwen CPU control

```text
prefix=64, suffix=8, cycles=16
clean same-seq max_abs=0
clean cross-seq max_abs=0
strict fork comparison PASS
fork metadata avg=0.007 ms
clean suffix=0.533–0.547 ms
fork suffix avg=0.519 ms
source state preserved (207,196 bytes)
```

### 35B one-GPU short-context control

```text
prefix=481, suffix=18, cycles=100
clean same-seq repeat is bitwise identical
cross-sequence clean variability: max_abs=0.504407, RMS=0.090417
fork comparisons remain within 1.5x measured clean variability
argmax token=357 for clean and forked paths
fork metadata avg=0.057 ms, max=0.069 ms
clean suffix≈52.2 ms
fork suffix avg=52.47 ms, max=53.58 ms
source state preserved (75,725,184 bytes)
```

FA-off testing produced comparable cross-sequence spread, so this variability is not specific to flash attention. Raw serialized source/fork buffers also differ because the format includes sequence IDs and physical KV/recurrent cell layout; same-sequence before/after state remains byte-identical and is the valid preservation check.

### 35B one-GPU 16k-prefix control

```text
prefix=16,001, suffix=18, cycles=16, unified KV
clean same-seq repeat is bitwise identical
coexisting clean A/B variability: max_abs=1.02972, RMS=0.19478
fork comparisons PASS against measured clean variability
argmax token=271 for all paths
fork metadata avg=2.060 ms, max=2.237 ms
clean suffix=60.56–67.82 ms
fork suffix avg=57.65 ms, max=59.82 ms
source state preserved (393,885,184 bytes)
```

### 35B four-GPU TP

Short-context, 32 cycles:

```text
fork metadata avg=0.057 ms
clean suffix≈73.2 ms
fork suffix avg=75.24 ms
source state preserved
PASS
```

Vocabulary-sharded output, 16 cycles:

```text
fork metadata avg=0.058 ms
fork suffix avg=76.82 ms
PASS
```

16k prefix, eight cycles:

```text
fork metadata avg=1.972 ms, max=2.297 ms
clean suffix=81.47–83.94 ms
fork suffix avg=74.93 ms (one 134.32 ms warmup/recapture outlier)
argmax token matches one-GPU reference (271)
source state preserved (393,885,184 bytes)
PASS
```

Source inspection findings:

- `examples/parallel`, `examples/speculative`, `examples/lookahead`, and batched tools already use `llama_memory_seq_cp()`.
- `llama_memory_hybrid::seq_cp()` forwards to both attention and recurrent memory.
- recurrent `seq_cp()` shares the source cell with the destination;
- recurrent `find_slot()` performs copy-on-write when the fork is first modified;
- existing tests cover recurrent rollback and host/device state migration, but not repeated direct fork-vs-clean logits for a hybrid model.