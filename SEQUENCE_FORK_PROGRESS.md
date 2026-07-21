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

- [ ] Load model once with a short context and at least two sequence IDs.
- [ ] Evaluate deterministic prefix on sequence 0.
- [ ] Fork sequence 0 to sequence 1 with `llama_memory_seq_cp()`.
- [ ] Continue both sequences with different suffixes.
- [ ] Compare each fork against clean recomputation.
- [ ] Compare full logits within a documented tolerance.
- [ ] Verify sampled tokens are identical.
- [ ] Measure fork and first-write copy-on-write latency.
- [ ] Repeat fork/continue/release cycles and check memory reuse.

### Gate 2 — Standalone feature ladder

- [ ] Tile FA on one GPU.
- [ ] HIP graphs.
- [ ] MTP target/draft state.
- [ ] Longer prefixes and varied suffix sizes.
- [ ] Unified KV.

### Gate 3 — 35B four-GPU TP

- [ ] Repeat deterministic clean-vs-fork fixtures.
- [ ] Compare TP logits/tokens with one-GPU reference.
- [ ] Verify sequence state is consistent on every rank.
- [ ] Measure fork overhead and PP throughput.
- [ ] Stress repeated forks at medium/long context.

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
- [ ] Inspect existing sequence-copy tests/examples and choose the smallest harness integration point.
- [ ] Implement Gate 1 harness.

## Decision Log

- Do not modify `llama-server` until standalone sequence-fork correctness is proven.
- Use 35B for iteration speed; use 122B only for targeted acceptance.
- Start on one GPU without TP, MTP, graphs, or unified KV.
- Add one feature at a time; do not use a combinatorial test matrix.
- Keep production and experimental builds isolated.

## Test Results

No sequence-fork harness results yet.