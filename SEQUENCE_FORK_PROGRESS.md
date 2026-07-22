# Sequence Fork / Turn-Boundary Snapshot Progress

Last updated: 2026-07-20

## Current Handoff

```text
Production branch: exp-gpu-sampling @ 1a3578dd6 (safe AMD checkpoint disable)
Experimental branch: exp-sequence-fork
Feature source commit: aea3814b8 (bounded shadow rollback)
Latest recorded result before this checkpoint: 60dd1912b
Experimental worktree: ~/llama-cpp-sequence-fork
Progress file: ~/llama-cpp-sequence-fork/SEQUENCE_FORK_PROGRESS.md
```

Current experimental requirements:

```text
GGML_SERVER_SEQUENCE_FORK=1
GGML_CUDA_DISABLE_GRAPHS=1
--kv-unified
--spec-type draft-mtp
--flash-attn on
```

Stateful restore permits exact matches and bounded shadow rollback within the recurrent snapshot window (`shadow_rollback <= n_rs`, currently 3). Larger mismatches discard the shadow before clean reprocessing. Internal active+shadow sequence capacity is allocated at target/draft context creation. Idle-slot cache eviction is disabled while the feature is active. CPU invariants validate active/shadow target/draft positions.

Post-audit staged results:

```text
35B single GPU exact shadow: PASS
35B TP exact shadow + MTP + vocab: PASS
35B captured 0–7: PASS
35B four concurrent exact shadows, mirrored output: PASS
122B captured 0–2: PASS
122B captured 0–7: PASS
122B captured 0–12 including deep clean reprocess: PASS
122B captured 0–15 exact-only policy: PASS
mmproj text-only and repeated-image lifecycle: PASS
all completed stages returned VRAM to idle
```

Concurrent dense grammar with `GGML_TP_VOCAB_OUTPUT=1` is fixed by serializing dense-incompatible slot output groups, eagerly accumulating dense rows while graph tensors are valid, and disabling MTP only for those slots. The compact-compatible parallel/MTP path remains enabled and has a passing regression.

Subagent status: a requested three-agent read-only audit on 2026-07-21 failed before launch because the local `pi-subagents` runtime could not resolve `typebox/compile`. No child modified files or produced findings.

## Current Diagnostic Breakpoint

Proven:

1. 122B requests 30–33 reproduce the illegal access in about 7.5 minutes with sequence forks enabled.
2. The same requests 30–33 complete without faults when sequence forks are disabled, although clean prompt processing takes about 21.5 minutes.
3. The enabled run faults during final-request prompt processing after an exact restore at 59,126 tokens; the request has 62,781 input tokens.
4. HIP reports the error at `ggml_backend_cuda_comm_vocab_top_k` during D2H synchronization. Because execution is asynchronous, this is an observation point, not yet proof that TOP_K caused the illegal access.
5. The failed `AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3` attempt never passed the script's 300-second readiness timeout. It produced no fault localization and must not be treated as evidence.
6. Every failed run cleaned up: no server/KFD process, VRAM returned to ~17 MiB per GPU, SMI remained responsive.

Unknown:

- whether the originating kernel is vocabulary TOP_K, long-context FlashAttention, unified-KV physical-span handling, or another asynchronous producer;
- whether vocabulary sharding is necessary for the reproducer;
- whether a single restored shadow at ~59k is sufficient, independent of earlier history.

Next focused work — do not start another long matrix:

1. create a 30–33 reproducer script with a configurable server-ready timeout and feature toggles;
2. add explicit stream synchronization/error checks immediately after candidate kernels in a diagnostic build so the first failing operation is reported;
3. run one fork-enabled 30–33 diagnostic reproduction;
4. only after localization, run the single discriminating toggle (for example vocab sharding off if TOP_K remains implicated);
5. after any GPU fault/timeout, stop and verify process/KFD/VRAM cleanup before another run.

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
- [x] MTP target/draft state, including copied per-sequence `pending_h`.
- [x] 16k prefix on one GPU.
- [ ] Varied suffix sizes.
- [x] Unified KV through a 64k fork/switch-back boundary.

### Gate 3 — 35B four-GPU TP

- [x] Repeat deterministic clean-vs-fork fixtures.
- [x] Verify TP argmax tokens match one-GPU reference.
- [x] Verify source sequence state remains byte-for-byte unchanged.
- [x] Measure fork overhead and suffix decode throughput.
- [x] Run repeated forks at a 16k prefix.
- [x] Verify vocabulary-sharded output remains compatible.
- [ ] Stress varied suffix sizes at longer context.
- [x] Re-run TP 16k switch-back after GPU reset; all marked phases passed with FA enabled.
- [x] Pass TP switch-back at 32k and 64k boundaries.

### Gate 4 — Targeted 122B confirmation

- [x] Basic clean-vs-fork correctness.
- [x] Repeated fork lifecycle.
- [x] 16k-prefix branch test.
- [x] MTP state alignment.

### Gate 5 — Server integration

- [x] Internal sequence-ID pool / shadow sequence mapping (IDs 0–3 active, 4–7 shadow).
- [x] Turn-boundary target/draft/MTP snapshot.
- [x] Shadow-to-active switch-back with bounded recurrent rollback.
- [x] Cancellation during prompt and generation; recovery request completed.
- [x] Shadow invalidation guards for LoRA/aLoRA changes, context shift, slot restore/erase, prompt clear, and child state copy.
- [x] `n_cmpl=2` parent/child target+draft+MTP copy and independent shadow snapshots.
- [x] Sleep/wake reload test: contexts/shadows destroyed, internal capacity recreated, next request starts initial.
- [ ] LoRA runtime test remains; no compatible adapter fixture is installed.
- [ ] Hard driver-timeout recovery is external; no automatic retry after GPU wedge.
- [x] Pi token-transition audit (offline usage/LCP proxy analysis).
- [x] Token-only dry-run planner.
- [x] Four parallel slots.
- [x] Unified-KV basic contention.
- [x] First 16 captured Pi requests on 35B.
- [ ] Full captured sessions and long-context server stress.
- [ ] 122B full-replay fault isolated to vocabulary-sharded TOP_K comm (`ggml_backend_cuda_comm_vocab_top_k`, hipMemcpyAsync D2H of gathered candidates) at ~62k context. Not caused by the fork lifecycle: fault followed a shadow-exact restore and 30 prior restores succeeded. Next reproducer: vocab top-k at ≥59k context without sequence forks.
  - 35B full 34-request replay passes with graph replay disabled.
  - 122B first 16 requests pass, including one deep-branch clean reprocess.
  - 122B full replay stalled after a two-token shadow rollback; stateful mode is being restricted to exact shadow matches because MTP `pending_h` has no rollback snapshots.
  - Full 34-request 35B replay reached 12 exact shadow restores, then page-faulted in tile FA on the 13th around 27k.
  - This used no host checkpoint load or rollback; repeated switch-back is now the focused standalone reproducer target.
- [x] Multimodal projector loaded for text-only exact shadow restore.
- [x] Same-image exact shadow restore lifecycle completed without faults.
- [ ] Image embeddings are still recomputed after restore; shadow memory currently preserves model sequence state, not cached mmproj chunk embeddings.
- [x] Concurrent dense grammar with `GGML_TP_VOCAB_OUTPUT`: dense slot output groups are serialized, dense rows are eagerly accumulated while graph tensors are valid, and MTP drafting is disabled only for dense-incompatible slots. Compact-compatible slots retain the original fast path.

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
- [x] Add deterministic/statistical multi-token continuation comparison.
- [x] Validate direct source-vs-fork continuation against clean cross-sequence variability.
- [x] Add MTP target/draft fork-state validation and versioned `pending_h` state hooks.
- [x] Validate shadow switch-back at 481, 16k, 32k, and 64k boundaries.
- [x] Pass targeted 122B basic, 16k, and MTP gates.
- [x] Complete offline Pi transition audit.
- [x] Design minimal server shadow-sequence lifecycle and exact boundary-selection policy.
- [x] Implement feature-gated token-only dry-run planner (`GGML_SERVER_SEQUENCE_FORK_DRY_RUN=1`) with no sequence-state mutation.
- [x] Replay focused request sequences and measure append/rollback/shadow/reset coverage.
  - Synthetic dry run: initial, shadow-bounded-rollback, active-append.
  - Captured requests 0–15: 15/15 follow-ups shadow-exact.
- [x] Implement opt-in stateful server prototype (`GGML_SERVER_SEQUENCE_FORK=1`).
- [x] Validate MTP + vocabulary sharding + TP shadow restore.
- [x] Validate four concurrent active/shadow pairs.
- [x] Add cancellation/error cleanup tests.
- [x] Full 35B captured-session replay with graph replay disabled.
- [x] Staged post-audit validation: 35B single GPU, 35B TP exact shadow, captured requests 0–7, four-slot mirrored-output exact shadows, and 122B requests 0–2.
- [x] Staged 122B requests 0–7 and 0–12 after lifecycle audit.
- [x] Staged 122B requests 0–15 with exact-only shadows; deep/bounded branches safely reprocessed and VRAM fully released.
- [ ] Full 122B replay remains deferred until the expensive repeated clean-reprocess path is improved or bounded MTP state is proven.

## Decision Log

- Do not modify `llama-server` until standalone sequence-fork correctness is proven. This gate is now satisfied through 64k TP and targeted 122B confirmation.
- Use 35B for iteration speed; use 122B only for targeted acceptance.
- Start on one GPU without TP, MTP, graphs, or unified KV.
- Add one feature at a time; do not use a combinatorial test matrix.
- Keep production and experimental builds isolated.
- Validate the server boundary policy in token-only dry-run mode before allocating internal shadow sequence IDs. Completed: one latest boundary covered all 15 captured follow-ups.
- Initial stateful mode requires unified KV and MTP, reserves one shadow ID per user slot, and disables idle-slot prompt-cache eviction so GPU shadows remain resident.
- Any operation that changes active token/state semantics (LoRA set changes, context shift, slot restore, slot erase, prompt clear) invalidates and clears the shadow first. `n_cmpl` child copies include speculative MTP state and clear old child shadows.
- Initial stateful mode also requires `GGML_CUDA_DISABLE_GRAPHS=1`. FA remains enabled. Full captured replay passed only when runtime HIP graph replay was disabled; performance was effectively unchanged in the measured workload.
- Internal sequence capacity must be allocated when target/draft contexts are created. Post-construction recurrent-only expansion was removed from the prototype.
- An unmatched shadow must be discarded before full reprocessing; otherwise unified KV retains the old prefix while allocating a duplicate active prefix, inflating physical FA span and memory pressure.
- Snapshot and restore positions are validated on CPU across target and draft memories before the next GPU decode.
- Existing single-slot prompt-cache shrink/expand logic is bypassed in sequence-fork mode so it cannot reduce preallocated internal sequence capacity.
- Stateful MTP shadow restore now allows bounded rollback (`shadow_rollback <= n_rs`). Root cause of the recurring two-token mismatch: the shadow boundary snapshot ends with the assistant `<think>\n` opener tokens, which the next captured request replaces with response text. Correctness is preserved because MTP drafts are always verified by the target model; a rollback-stale `pending_h` can only reduce draft acceptance, never change output. Target/draft memory rolls back via `llama_memory_seq_rm` within recurrent snapshot range. The earlier standalone rollback experiment was chasing draft-identity, which is not required.

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

16k prefix, statistical continuation:

```text
fork metadata avg=1.923 ms, max=2.213 ms
clean suffix=80.70–83.65 ms
fork suffix avg=70.31 ms (one 133.73 ms warmup/recapture outlier)
16-token clean spread: max_abs=0.99656, RMS=0.200011, mismatches=1
16-token fork spread: max_abs=1.08603, RMS=0.203496, mismatches=1
16-token direct source/fork: max_abs=1.33537, RMS=0.210420, mismatches=1
all fork/direct spread remains below 1.5x clean-layout bound
source state preserved (393,885,184 bytes)
PASS
```

### Multi-token correctness interpretation

The tiny CPU model is bitwise exact over 16 continuation tokens for clean, fork-vs-clean, and direct source-vs-fork paths. The quantized 35B GPU model is not sequence-layout deterministic: independently clean coexisting sequences can select different greedy tokens under the same token history. Forked and direct-source continuations show comparable mismatch counts and logit spread, bounded against measured clean-layout variability. Same-sequence recomputation remains bitwise identical.

At 16k on one GPU:

```text
clean spread max_abs=1.35186, RMS=0.232532, mismatches=1/16
fork spread max_abs=1.19464, RMS=0.230882, mismatches=0/16
direct source/fork max_abs=1.10825, RMS=0.230374, mismatches=1/16
fork/direct spread below 1.5x clean-layout bound
PASS
```

At 16k on four-GPU TP:

```text
clean spread max_abs=0.99656, RMS=0.200011, mismatches=1/16
fork spread max_abs=1.08603, RMS=0.203496, mismatches=1/16
direct source/fork max_abs=1.33537, RMS=0.210420, mismatches=1/16
fork/direct spread below 1.5x clean-layout bound
PASS
```

### MTP fork-state validation

Qwen MTP maintains per-sequence `pending_h` carryover in `common_speculative_impl_draft_mtp`. The generic speculative `get_state`/`set_state` APIs previously had no MTP override. The experimental branch now serializes a versioned per-sequence `pending_h` row.

Results on 35B:

```text
single GPU, 241-token prompt: PASS
four-GPU TP, 241-token prompt: PASS
four-GPU TP + vocabulary sharding: PASS
single GPU, 16,001-token prompt: PASS
four-GPU TP, 16,001-token prompt: PASS
MTP pending state size: 8,200 bytes
source draft == copied-fork draft (3 tokens)
uncopied negative-control state/draft differs
```

A sequence fork must therefore copy all three components:

1. target memory via `llama_memory_seq_cp`;
2. draft memory via `llama_memory_seq_cp`;
3. MTP `pending_h` via speculative state get/set.

Existing recurrent rollback regression test remains clean after the MTP state-hook change.

### Shadow switch-back and scale validation

The standalone harness now preserves a boundary shadow, continues the active sequence down an unrelated branch, clears the active sequence, copies the shadow back into the same active sequence ID, and evaluates the canonical branch with tile FA.

Results:

```text
tiny CPU: bitwise exact switch-back PASS
35B one GPU, 481-token boundary: PASS
35B one GPU, 16k boundary: PASS
35B TP, 16k boundary after GPU reset: PASS
35B TP, 32k boundary: PASS
35B TP, 64k boundary with 262k unified pool: PASS
```

64k TP details:

```text
prefix=64,001
fork metadata avg=9.133 ms, max=9.498 ms
clean suffix=136.50–182.50 ms
fork suffix avg=109.68 ms
shadow switch-back max_abs=0.73944, RMS=0.143615, argmax matched
source state preserved (1,377,885,184 bytes)
```

One pre-reset TP 16k rerun page-faulted during ordinary source-prefix tile FA before any `seq_cp`. After resetting the GPUs, the exact marked fixture passed all phases. This was residual GPU state from the earlier driver timeout, not a fork failure.

### 122B targeted confirmation

```text
basic TP, eight fork cycles: PASS
fork metadata avg=0.068 ms
fork suffix≈120.66 ms vs clean≈116.9 ms
8-token clean/fork/direct continuations: zero mismatches
source state preserved (168,135,232 bytes)
16k TP switch-back: PASS
16k fork metadata avg=1.982 ms
16k source state preserved (549,865,152 bytes)
MTP state=12,296 bytes; copied draft matches source; negative control differs
```

### Offline Pi transition audit

Two captured sessions show median prompt-prefix reuse of 96–98%. Approximately 36% of turns add at most 256 uncached tokens; many larger turns still retain nearly the entire prior prefix. This validates turn-boundary forks as the correct optimization target.

### Server dry-run and stateful prototype

Synthetic three-request planner:

```text
request 1: initial
request 2: shadow-bounded-rollback (one assistant-marker token)
request 3: active-append
```

Captured Pi requests 0–15 dry run:

```text
initial=1
shadow-exact=15
full reprocess after initialization=0
```

Captured Pi requests 0–15 stateful run:

```text
shadow restores=15
snapshots=16
faults/errors=0
safe baseline: 320,496 prompt tokens, 248.8 s PP
stateful fork: 31,481 prompt tokens, 24.5 s PP
reduction: 10.2x in evaluated prompt tokens and prompt time
```

Four-slot concurrent stateful run:

```text
4 initial active/shadow snapshots
4/4 shadow-bounded restores
4/4 subsequent active appends
12/12 requests
8,272 total evaluated prompt tokens
faults/errors=0
```

Full captured-session stateful replay with graph replay enabled:

```text
12 exact shadow restores completed
13th exact shadow restore page-faulted in tile FA around 27k
no host checkpoint load
no recurrent rollback
failure surfaced later during vocab candidate transfer synchronization
```

Focused standalone controls all passed:

```text
32 repeated target-only switch-backs
32 repeated target+draft+MTP switch-backs
32 repeated MTP switch-backs with vocabulary sharding
32 repeated cycles with 401-token suffix and 512-token active tail
```

Full captured-session stateful replay with `GGML_CUDA_DISABLE_GRAPHS=1`:

```text
34/34 requests
33 exact shadow restores
66,770 total prompt tokens evaluated
59.8 s total prompt processing
0 faults/errors
```

Performance A/B for requests 0–15:

```text
graphs enabled:  PP 24.455 s, generation 82.719 s
graphs disabled: PP 24.486 s, generation 82.780 s
```

Conclusion: stale/replayed HIP graphs are a server-only failure multiplier after sequence-ID state switching. The first stable implementation requires graph replay off while retaining FA.

### 122B driver-timeout audit

A later 122B exact-match-only replay completed ordinary shadow restores, then selected full reprocessing for deep/bounded branches. The prototype incorrectly kept the unmatched shadow resident while rebuilding the entire active prompt. Under unified KV this duplicated tens of thousands of physical prefix cells, expanded the FA cache span, collapsed PP from roughly 544 t/s toward 238 t/s, and ended in a driver wedge with retained VRAM after process exit.

Safety changes made offline before any rerun:

```text
unmatched shadow is cleared before full reprocess
internal target/draft sequence capacity is allocated at context creation (8), not expanded after scheduler initialization
stateful MTP restore is exact-shadow-only
runtime HIP graph replay remains disabled
CPU invariants verify target/draft active and shadow positions before decode
```

Unified KV idle-slot prompt-cache eviction cleared shadow lifecycle state in the first parallel attempt. Experimental stateful mode now disables idle-slot eviction so shadows remain GPU-resident. Attention prefix cells remain shared; recurrent state uses copy-on-write.

### Staged post-audit validation

```text
35B single GPU, one exact shadow: PASS
35B TP + MTP + vocab, one exact shadow: PASS
35B captured requests 0–7: 8/8, seven exact shadows, PASS
35B four concurrent exact shadows, mirrored output: 8/8, PASS
122B captured requests 0–2: 3/3, one exact shadow, PASS
122B captured requests 0–7: 8/8, six exact shadows, PASS
122B captured requests 0–12: 13/13, unmatched shadow discarded before deep clean reprocess, PASS
122B captured requests 0–15: 16/16, ten exact shadows and five safe clean reprocess decisions, PASS
122B captured requests 0–20: 21/21, eleven exact shadows, nine discarded shadows with clean reprocess, 0 faults, PASS
122B captured requests 0–15 with bounded rollback: 16/16, ten exact + four bounded-rollback (2-token) restores, one full reprocess, PP 56,806 tokens/81.0 s vs 151,158/398.6 s exact-only (4.9×), decode speed unchanged, 0 faults, PASS
122B captured requests 0–20 with bounded rollback: 21/21, eleven exact + eight bounded-rollback restores, one full reprocess (vs nine exact-only), PP 65,056 tokens/118.4 s vs 290,811/842.4 s (7.1×), wall 8:50 vs 20:23, 0 faults, VRAM fully released, PASS
122B full replay (0–33) with bounded rollback: 33/34 completed, 30 shadow restores (22 exact + 8 bounded), 3 safe full reprocesses, then illegal memory access in ggml_backend_cuda_comm_vocab_top_k at ~62k tokens on the final request. Host exited cleanly: no wedge, no KFD process, VRAM released, SMI responsive.
35B + real mmproj, text-only exact shadow: PASS
35B + real mmproj, repeated identical image prompt: lifecycle PASS (image chunk still recomputed)
35B single-GPU sleep/wake reload: PASS; no stale shadow after reload
35B TP `n_cmpl=2`: two choices, parent/child shadows created, PASS
all stages returned VRAM to idle and left no KFD process
```

A four-concurrent tool/grammar workload with vocabulary sharding originally asserted in dense-logit materialization even when sequence-fork mode was disabled. The fix now combines:

1. sampler capability detection before decode;
2. serialization of output groups when any involved slot requires dense logits;
3. eager accumulation of dense rows while each graph tensor remains valid;
4. MTP drafting disabled only for dense-incompatible slots so speculative output groups never cross serialized views;
5. bounds-checked lazy fallback as a fail-safe.

Validation:

```text
four concurrent tool/grammar requests, no sequence fork: 8/8 PASS
four concurrent exact sequence-fork shadows + vocab sharding + grammar: 8/8 PASS
0 faults/errors; VRAM fully released
```

Dense concurrent generation is intentionally slower because it is serialized and does not use MTP drafting. Compact-compatible vocabulary-sharded slots retain parallel/MTP fast paths.

Compact-path regression after the dense fix:

```text
four parallel compact-compatible vocab-sharded lanes: PASS
batched decode preserved; MTP drafting active
draft acceptance 0.78–0.81, mean draft length ~3.4
0 faults/errors; VRAM fully released
```

Deep-branch safety result on 122B:

```text
unmatched shadow discarded before full reprocess
26,917-token clean reprocess: 38.4 s, 701.7 t/s
0 faults; VRAM fully released
```

Later bounded shadow mismatches remain exact-only fallbacks. Repeated full reprocesses are stable but slow as unified-KV physical span/fragmentation grows; bounded MTP rollback is not enabled.

122B requests 0–20 confirmed a recurring pattern: the deep-branch turn repeatedly left a two-token shadow mismatch (shadow_rollback=2), forcing a full clean reprocess every other turn beyond ~28k tokens. Token-level diagnosis showed the mismatch is always the assistant `<think>\n` opener at the snapshot boundary. Bounded shadow rollback (≤ n_rs) now handles it: MTP drafts remain target-verified so stale `pending_h` is safe, and target/draft memory rollback uses existing recurrent snapshots. Requests 0–15 dropped from six full reprocesses to one.

Source inspection findings:

- `examples/parallel`, `examples/speculative`, `examples/lookahead`, and batched tools already use `llama_memory_seq_cp()`.
- `llama_memory_hybrid::seq_cp()` forwards to both attention and recurrent memory.
- recurrent `seq_cp()` shares the source cell with the destination;
- recurrent `find_slot()` performs copy-on-write when the fork is first modified;
- existing tests cover recurrent rollback and host/device state migration, but not repeated direct fork-vs-clean logits for a hybrid model.