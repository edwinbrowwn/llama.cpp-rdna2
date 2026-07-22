# Sequence Fork / Turn-Boundary Snapshot Progress

Last updated: 2026-07-20

## Current Handoff

```text
Production branch: exp-gpu-sampling @ 1a3578dd6 (safe AMD checkpoint disable)
Experimental branch: exp-sequence-fork
Experimental safety head: f87b17ed9 (atomic full-reprocess transition; GPU-unvalidated)
Peer-gather ordering patch reverted: 049375fbe
Bounded-shadow source commit retained: aea3814b8
Previous diagnostic checkpoint: 0a4a268df
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

## STOP: Full-Replay GPU Timeout After Gather-Ordering Patch

The targeted requests 30–33 reproduction passed once with source patch `16565727c`, but the subsequent full 122B requests 0–33 validation caused the GPUs to time out and required the user to shut the server down manually. The command was aborted; no completion result is valid.

Recovered evidence and safety status:

- Host restarted cleanly: no server/KFD process, all GPUs at ~17 MiB, SMI responsive.
- Requests 0–11 completed. Request 12 (`26,917` input tokens) discarded the unmatched shadow and selected `full-reprocess`; the server log stops immediately after that plan.
- No prior-boot kernel entry matches this request-12 job PID (`40276`); the persisted GPU page faults belong to the earlier request-33 full run (`33581`) and focused reproduction (`34718`). Request 12 is therefore recorded as a GPU timeout/manual shutdown, not a proven page fault.
- In both proven request-33 failures, ROCm names `flash_attn_tile<256,256,4,8,false>` as the faulting kernel. TOP_K/D2H only reported the already-existing asynchronous FA error.
- Experimental patch `16565727c` was reverted by `049375fbe` before any further workload.
- Do **not** rerun the full replay. Production branch `exp-gpu-sampling @ 1a3578dd6` was not modified.
- Offline audit found that `decision=full-reprocess` discarded only the shadow and did not itself force `n_past=0` or clear the active sequence; it depended on later generic checkpoint heuristics. This violated the intended safe-fallback semantics and left shared cells live across the transition.
- Experimental fix `f87b17ed9` makes full reprocess atomic: synchronize target/draft, clear active and shadow state plus prompt checkpoints, set `n_past=n_past_common=0`, then rebuild. Server build, CPU sequence-fork cycles, and recurrent rollback tests pass.
- `f87b17ed9` has **not** been GPU-tested. Do not run another full replay without a separately approved, bounded validation plan.

## Page-Fault Triage: What Matters and What Does Not

### Established facts

- Production is safe and unchanged; the fault exists only on the experimental sequence-fork branch.
- Core sequence copy/switch correctness, recurrent rollback, 35B TP, MTP, four slots, cancellation, sleep/wake, multimodal lifecycle, and 122B requests 0–20 have passing staged results.
- Two reproducible request-33 failures are timing/history-sensitive: ROCm identifies the faulting kernel as tile FA near 62k after an exact restore; TOP_K/D2H is only the later synchronization point. A separate request-12 run with the experimental gather patch timed out without a persisted kernel record.
- The same requests 30–33 complete without sequence forks, so context length alone is not sufficient.
- The deep request had discarded its unmatched shadow, but the old planner had not atomically cleared active state or forced token-zero rebuild at the decision point.

### Captured replay evidence matrix

| Run | Range/configuration | Outcome | Key implication |
|---|---|---|---|
| 35B full, HIP graphs enabled | 0–33 | FA tile fault near request 12 / ~27k | graph replay was unsafe for forked FA |
| 35B full, graphs disabled | 0–33 | 34/34 pass through ~67k | current 35B control passes |
| 122B staged exact-only | 0–12, 0–15, 0–20 | all pass; request-12 full reprocess passes repeatedly | request 12/input length alone is not sufficient |
| 122B bounded rollback | 0–15, 0–20 | all pass | bounded rollback alone is not sufficient |
| 122B older full | through request 12 | request 12 full reprocess completes; request 13 bounded restore times out | state after full rebuild may poison successor transition |
| 122B bounded full | 0–33 | request 33 FA tile page fault | proven long exact-restore failure |
| 122B focused fork | 30–33 | request 33 FA tile page fault | earlier requests 0–29 are unnecessary |
| 122B focused no-fork | 30–33 | 4/4 pass | fork/restore history is necessary; length alone is not |
| 122B broad-sync diagnostic | 30–33 | pass | timing changes suppress the fault but do not localize it |
| 122B gather-order experiment | 30–33 pass; later full run times out at request 12 | experiment changes timing and is unsafe; reverted | not evidence for gather as root |

The passing and failing request-12 runs have essentially identical logical lengths/LCPs. The primary unknown is physical ownership/layout and completion ordering, not token selection.

### Earlier restored-state experiments

The pre-fork checkpoint investigation already isolated the same failure class on 122B:

```text
checkpoint restore + vocab output: crash
checkpoint restore + mirrored output: crash
checkpoint restore without MTP: crash
checkpoint restore with HIP graphs disabled: crash
checkpoint restore with device-context/synchronization workaround: crash
skip checkpoint load entirely: pass
checkpoints disabled: pass
minimal requests 26–27 with restored tile FA: crash
vector FA for only the first restored microbatch: later crash
vector FA for the entire restored suffix: pass
high-context full 34-request 122B replay with full-suffix vector FA: pass
```

Key artifacts:

- `~/llama-jobs/checkpoint-isolation-summary/results.txt`
- `~/llama-jobs/replay-session-019f81d5-high-context-vector`
- `~/llama-jobs/replay-session-019f81d5-near-limit-vector`
- `~/llama-jobs/replay-min-26-27-fa-vec*`
- `~/llama-jobs/replay-min-26-27-full-checkpoint`

The full-suffix vector replay completed all 34 requests; its final 3,659 restored prompt tokens ran at 38.5 t/s without a fault. The near-limit first-microbatch-only variant later returned to tile FA and faulted. This establishes that graph reset or one vector microbatch is insufficient, while avoiding tile FA for the entire restored suffix is a known-safe precedent.

Sequence forks avoid host checkpoint serialization and tensor copies, but they still present restored sequence state to FA. The identical tile kernel fault means the checkpoint evidence transfers directly to server policy even if the ultimate backend defect remains unknown.

### Ruled out or deprioritized

- **Concurrent dense vocabulary grammar bug:** separately fixed and independently regressed; not the current blocker.
- **TOP_K buffer sizing / peer-gather ordering as root cause:** not supported. TOP_K was an asynchronous observation point; the ordering experiment is reverted.
- **HIP graph replay:** explicitly disabled in failing runs.
- **AMD host checkpoint restore:** disabled; no host checkpoint load occurred.
- **Generic 62k model limit:** no-fork requests 30–33 completed at the same length.
- **Multimodal, LoRA, and multi-slot lifecycle:** absent from the failing single-slot captured workload.
- **Persistent leak after failure:** not the origin; restart returned all GPUs/KFD state to idle.

### Ranked live hypotheses

1. **AMD tile FA is unsafe on restored hybrid/recurrent prompt state.** This is directly supported by both checkpoint and sequence-fork evidence. The exact backend defect may be physical KV layout, mask/view lifetime, or asynchronous pool reuse, but restored tile dispatch is the proven common trigger.
2. **Non-atomic shared-cell transition at full reprocess.** Target/draft operations were not synchronized and active state was not immediately cleared. This can poison the next restored shadow; `f87b17ed9` remains sensible hardening for clean fallback but is not sufficient to make restored tile FA supported.
3. **Unified-KV physical-cell reuse / FA tensor view after active+shadow history.** This is the leading backend root-cause family if tile FA itself must be repaired.
4. **Cross-device asynchronous KV writes or temporary-buffer lifetime.** Timing-sensitive sync results keep this plausible, but they do not justify broad barriers.
5. **Bounded rollback history.** Lower priority because exact restores and checkpoint restores also fault.
6. **Vocabulary sharding.** Ruled out as primary by mirrored/no-MTP checkpoint crashes and the explicit FA faulting kernel.

### Recommended completion architecture

Treat restored-prompt tile FA as unsupported on AMD hybrid/recurrent models until the backend is independently repaired:

- initial/append-only prompts: normal tile FA;
- clean full reprocess: atomic reset plus normal tile FA;
- exact or bounded shadow restore with a small suffix (initial conservative threshold: suffix ≤ 1/8 of full prompt): vector FA for every remaining prompt token;
- larger restored suffix: reject restore and use atomic clean tile-FA full reprocess, which is both safer and faster at that ratio;
- after the restored suffix is consumed: return to normal kernel selection for generation;
- if vector FA is not eligible: reject the restore and perform a clean tile-FA full reprocess;
- keep `--flash-attn on` for TP and keep HIP graph replay disabled for the experimental sequence-fork mode. Vector FA is an internal FA kernel selection, not `--flash-attn off`.

This preserves the main benefit: the final failing request evaluates ~3.7k restored suffix tokens instead of ~62.8k clean tokens. It sacrifices tile speed only on the reused suffix. Prior full-suffix vector evidence measured 38.5 t/s for the final suffix and completed the full 122B session.

Retained optimizations: four-GPU tensor splitting, FA enabled, unified KV, MTP, NCCL/RCCL all-reduce, vocabulary-sharded compact TOP_K, dense-grammar serialization/fallback, and compact-compatible parallel batching. Restored/tile batch separation is an additional temporary split only while restored suffix tokens remain; unrelated compact slots retain their fast path.

Measured restored-suffix cost at the final request:

```text
tile suffix (diagnostic passing runs): 155–173 t/s, ~21–24 s for 3,655 tokens
full-suffix vector:                     38.5 t/s, ~95 s for 3,659 tokens
clean full prompt:                     231 t/s, ~272 s for 62,781 tokens
```

Thus vector is ~4.0–4.5× slower than tile for those suffix tokens (about 75–78% lower suffix PP throughput), but still ~2.9× faster than clean full reprocessing for the complete request and saves about 177 seconds of PP. The theoretical measured break-even is near suffix/full ≈ 1/6; use 1/8 initially for margin.

### Restored-suffix vector policy considerations

**Transitions that mark a suffix as restored**

- shadow-exact restore;
- shadow-bounded rollback;
- active bounded rollback;
- any future host/device state restore.

Initial, active-append, and atomic full reprocess remain tile-FA paths.

**Counter semantics**

- Keep `restored_prompt_tokens_remaining` per slot/sequence, never one global counter.
- Set it to the exact number of prompt tokens that will be decoded after the restored boundary, including any mandatory one-token prompt-logit reevaluation.
- Decrement only by prompt tokens actually decoded for that sequence, not by total mixed-batch or MTP draft tokens.
- Clear it on completion, cancellation, slot release/purge, prompt clear, LoRA change, context shift, sleep/wake, restore failure, and clean full reprocess.

**Batching/concurrency**

- Do not let a process-global kernel-selection flag leak across unrelated slots.
- Split target decode views at restored/non-restored slot boundaries before setting the force-vector mode.
- Preserve speculative output-group integrity; if a safe split is impossible, disable drafting for that restored view rather than crossing the boundary.
- It is safe for a mixed view to use vector for all tokens, but that is an explicit conservative fallback, not the default fast path.

**Backend selection**

- Prefer a scoped context/decode flag carried into FA kernel selection over `setenv()/unsetenv()`.
- If the old environment mechanism is temporarily reused, prove the server decode loop is single-threaded and set/unset it with RAII around exactly one `llama_decode` view.
- Keep HIP graph replay disabled; a vector selection must never reuse a cached tile graph.
- Check vector eligibility before restore. If head/type/layout is unsupported, abandon restore and use atomic clean tile reprocess.

**MTP and correctness**

- Target vector FA changes only verification/prompt evaluation; MTP drafts remain proposals and target verification remains authoritative.
- Previous full-suffix vector runs included 122B MTP, but add deterministic logit/argmax comparison for the sequence-fork path.
- Record MTP acceptance separately; lower acceptance is a performance issue, not a correctness failure.

**Multimodal and lifecycle**

- A restored image-containing prompt follows the same target policy; if vector eligibility or batch splitting is ambiguous, clean reprocess.
- Child `n_cmpl` slots must not inherit stale restored-token counters.
- Sleep/wake and model reload recreate counters at zero.

**Acceptance**

- Logs must state restore boundary, exact vector token count, kernel-policy entry/exit, and fallback reason.
- No tile FA launch may occur for a marked restored target suffix.
- Counter must reach zero exactly when the restored suffix ends.
- Generation and later clean prompts return to normal kernel selection.
- Cancellation at every restored ubatch boundary leaves no marked slot or sequence state.

### Strict next sequence

1. Design a per-slot restored-suffix counter and batch-splitting policy; avoid a process-global flag leaking to unrelated slots.
2. Reintroduce vector-kernel selection in a scoped backend/context mechanism, or use the old environment mechanism only if the server decode loop proves single-threaded and the flag is set/unset around exactly one batch view.
3. Add CPU tests for counter accounting across multiple ubatches, bounded restore, exact restore, cancellation, and mixed restored/non-restored slots.
4. Retain host ownership/layout invariants and atomic clean fallback; do not use their absence as permission for restored tile FA.
5. GPU ladder only after code review: 35B small restore, 122B requests 26–27/minimal restored suffix, 122B requests 30–33, then one full replay.
6. Keep backend tile-FA root-cause instrumentation as a separate research track; it is not required to ship a safe sequence-fork policy.

## Pre-GPU Logical Validation Plan

### Failure modes must remain separate

**Scenario A — long exact restore**

```text
shadow exact at 59,126
input 62,781
incremental suffix 3,655
fault surfaced near the end of prompt processing at TOP_K/D2H
kernel journal: GPU TCP read page fault
```

**Scenario B — deep full reprocess timeout**

```text
active 27,268; active LCP 11,538
shadow 26,862; shadow LCP 11,538
shadow discarded; input 26,917
server stopped immediately after full-reprocess plan
manual shutdown; no matching persisted kernel fault for PID 40276
```

Scenario A is immediately preceded by request 32 doing a 59,126-token full reprocess. Therefore `f87b17ed9` can potentially explain both: directly hardening B and preventing request 32 from creating a noncanonical layout later consumed by A. This is a hypothesis, not yet validation.

### State ownership model to validate

For each user slot, track these independently:

1. target attention active sequence;
2. target recurrent active sequence and rollback index;
3. draft attention active sequence;
4. draft recurrent active sequence and rollback index;
5. target/draft shadow counterparts;
6. MTP `pending_h` / speculative state for active and shadow IDs;
7. server prompt tokens and checkpoints;
8. unified-KV physical stream: head, used cells, highest used cell, per-sequence cell/ref counts;
9. graph inputs: selected KV indices and resulting `n_kv`/physical span.

Logical position equality is necessary but insufficient: hybrid `seq_pos_max()` can hide a residual component because it combines attention and recurrent ranges. Audit data must inspect attention and recurrent components separately.

### Required transition postconditions

**Snapshot**

- target and draft active contexts are complete before metadata mutation;
- active target/draft positions equal `boundary_tokens - 1`;
- shadow target/draft positions equal active after copy;
- attention copy shares cells without increasing physical cell count;
- recurrent source/destination references are valid;
- copied MTP state is associated with the same boundary.

**Exact restore**

- old active-only generation cells are released;
- copied active and resident shadow have the same logical boundary;
- no cell references an unexpected sequence ID;
- physical `used` and `used_max_p1` are bounded by the active/shadow union and do not grow solely from repeated restore;
- selected FA indices are within the backing cache and below computed `n_kv`.

**Bounded restore**

- all exact-restore conditions hold;
- target/draft position becomes `shadow_lcp - 1`;
- rollback is within recurrent snapshot capacity;
- stale MTP state is treated only as a proposal source and must be refreshed before it can become authoritative.

**Full reprocess**

- target and draft are synchronized before sequence metadata changes;
- active and shadow attention/recurrent references are all absent after clear;
- recurrent rollback indices reset;
- prompt tokens and checkpoints are empty;
- unified-KV head/used/span return to canonical empty values for the slot/stream when no other slot is active;
- first new token is position zero;
- no graph is built if any clear invariant fails.

### Instrumentation requirements before GPU use

Use one environment-gated host audit mode; do not add broad device synchronizations that change timing.

At each snapshot/restore/full-reset boundary log:

- operation, slot, active/shadow IDs;
- target/draft attention min/max, recurrent min/max, and recurrent rollback index;
- attention stream head, used cells, `used_max_p1`, and per-sequence ref counts;
- first/last selected KV cell for each ubatch and computed `n_kv`;
- prompt token/checkpoint counts and MTP state validity.

Fail closed before decode on an impossible sequence reference, nonempty full-reset component, out-of-range cell index, or `max_selected_index >= n_kv`.

### Hypothesis matrix and falsifiers

| Hypothesis | Explains A | Explains B | Evidence that would falsify it |
|---|---:|---:|---|
| Restored prompt state is unsafe on AMD tile FA | Yes | B is not a proven FA fault | a full-suffix vector policy faults in the vector kernel under the same restored trace; existing evidence instead shows full-vector pass |
| Non-atomic full reset / poisoned successor layout | Yes, indirectly via request 32 | Possibly | A fails despite canonical atomic-clear and post-rebuild layout while full-suffix vector remains safe |
| Unified-KV fragmentation / bad physical span | Yes | Possibly | canonical bounded head/used/span and valid indices immediately before restored tile fault |
| Async target/draft KV writes during metadata mutation | Yes | Possibly | explicit boundary synchronization plus canonical stats still produces the restored tile fault |
| Bounded rollback/MTP history | No for exact/checkpoint failures | Possibly | already deprioritized by exact and no-MTP checkpoint crashes |
| Vocabulary sharding | No as primary | No evidence | already deprioritized by mirrored checkpoint crashes and explicit FA kernel report |
| Generic model length/hardware | Weak | Weak | already weakened by no-fork 62k and full-vector 34-request passes |

### Inverted checks

Before accepting the current approach, answer:

- Why did old code sometimes pass Scenario B? A correct answer must include timing/layout differences, not merely the input tokens.
- Why did diagnostic synchronization make Scenario A pass? It may hide any upstream race; it does not implicate the synchronized function.
- Why did the gather patch pass A but fail earlier at B? This argues that changed timing, not gather correctness, selected the outcome.
- Why can A fail after an exact restore if full reset is the bug? Because its restored shadow was created immediately after request 32 rebuilt 59,126 tokens. Validate both the request-32 post-rebuild layout and request-33 exact-restore layout.
- What observation would make us revert `f87b17ed9`? Canonical old/new full-reset states with no successor-layout difference, or a new regression caused by forced synchronization/clear semantics.
- Are we solving the backend or shipping a safe policy? Full-suffix vector FA has direct full-session evidence; restored tile FA repair does not. Do not block safe completion on backend research unless tile-only operation remains a hard requirement.

### Safety and stop rules

- Never start with a full replay.
- No GPU run without clean process/KFD/VRAM precheck and user awareness.
- A timeout is not recovery; on page fault, stop immediately and require health verification or reboot.
- Preserve all logs and prior-boot journal evidence before any next run.
- Do not alter more than one causal dimension per discriminator.
- Do not promote experimental changes based only on absence of a fault.

## Long-Context Fault Investigation

Focused reproducer: `~/ar-bench/run-sequence-fork-long-context-diag.sh` defaults to captured requests 30–33 and exposes only readiness, fork, vocab-output, and diagnostic-sync toggles.

Proven sequence:

1. Fork-enabled requests 30–33 reproduced the illegal access deterministically on final-request prompt processing (exact restore at 59,126 tokens; input 62,781). ROCm identifies `flash_attn_tile<256,256,4,8,false>` as the faulting kernel; D2H in `ggml_backend_cuda_comm_vocab_top_k` is only where the asynchronous error is reported.
2. The same requests completed without forks, ruling out a generic 62k model/FA limit.
3. An environment-gated diagnostic bracket synchronized streams before local TOP_K, after local TOP_K, and after NCCL all-gather. The reproducer passed, proving only that the failure is timing/order-sensitive; it did not localize the originating operation.
4. Code inspection found that peer NCCL streams were synchronized only *after* rank 0 began reading the gathered buffer to host. On ROCm, rank 0 could reach D2H while peer all-gather work was still completing.
5. Experimental patch `16565727c` moved the three existing peer synchronizations before the D2H read and removed the broad diagnostic barriers.
6. One uninstrumented fork-enabled 30–33 run passed 4/4, including the 62,781-token final request.
7. The immediately following full 122B replay timed out at the request-12 deep full-reprocess transition and required manual shutdown. No matching kernel record persisted. This still disproved the claimed gather localization, but it must not be conflated with the two proven request-33 FA page faults.
8. `16565727c` is reverted by `049375fbe`. No GPU validation was run after the revert.

Offline source analysis produced atomic full-reprocess fix `f87b17ed9`. It is CPU-tested but deliberately GPU-unvalidated after the manual-shutdown incident. No further workload should be launched in this iteration.

## Goal

Replace in-place recurrent checkpoint restore on AMD with a turn-boundary sequence fork/switch lifecycle that:

- preserves prefix reuse for Qwen hybrid/recurrent models;
- keeps clean prompt processing on tile flash attention; restored AMD hybrid suffixes use full-suffix vector FA until the backend tile defect is independently resolved;
- produces the same logits as clean recomputation;
- works first on one GPU, then four-GPU tensor parallelism;
- supports MTP and eventually unified KV plus four server slots;
- avoids host checkpoint serialization on the common path;
- avoids full prompt reprocessing on the common path while confining the AMD vector-FA safety fallback to restored suffix tokens only.

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
- [ ] 122B GPU page fault remains unresolved. TOP_K was only an asynchronous observation point; a later full run page-faulted on a TCP read immediately after the request-12 deep full-reprocess plan.
  - 35B full 34-request replay passes with graph replay disabled.
  - 122B requests 0–20 previously passed with bounded rollback: 19 shadow restores, one clean reprocess, zero faults.
  - Peer-gather ordering experiment `16565727c` is reverted by `049375fbe`.
  - Atomic full-reprocess fix `f87b17ed9` is build/CPU-tested only; GPU validation remains explicitly deferred.
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