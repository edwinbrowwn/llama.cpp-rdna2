# TP2×PP2 versus TP4 on Qwen3.8 Q8

## Decision

The explicit four-GPU TP2×PP2 implementation is functionally valid, but it does **not** beat ordinary four-GPU TP4 for token generation with the same Qwen3.8-27B Q8 model. Keep it experimental and explicit opt-in. Do not present gains over two-GPU TP2 as a win over TP4.

This follow-up did not optimize TP4 and count that as hybrid success. Every causal comparison used:

```text
/home/edwin/models/qwen38-27b-q4s8/unsloth-q8/Qwen3.8-27B-Q8_0.gguf
SHA-256 a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348
```

The work is isolated on `perf/tp2-pp2-beat-tp4-q8`; `master` was not modified.

## Matched results

### PP512/TG128

| Concurrency | Topology | Prompt tok/s | Generation tok/s |
|---:|---|---:|---:|
| 1 | TP4 | 940.07 | 27.77 |
| 1 | TP2×PP2 | 734.62 | 23.59 |
| 4 | TP4 | 1175.26 | 94.63 |
| 4 | TP2×PP2 | **1210.38** | 78.65 |
| 16 | TP4 | 880.49 | 136.23 |
| 16 | TP2×PP2 | **924.00** | 116.18 |

Hybrid prompt processing can beat TP4 under concurrency. Generation remains 15–17% behind.

At PP128/TG64 C32, hybrid generation was `187.60 tok/s` versus TP4 at `235.85 tok/s` (`-20.46%`). There was no high-concurrency TG crossover.

### Backend sampling

Backend sampling removed much of the full-logit readback/sampling cost:

| C4 topology | Prompt tok/s | Generation tok/s |
|---|---:|---:|
| TP4 | 969.92 | 88.52 |
| TP2×PP2 | **1062.43** | 79.69 |

The generation deficit narrowed to `-9.98%`, but did not reverse.

## Why generation is slower

### Ordinary decode has no pipeline item

The server combines active sequences into one decode graph per autoregressive step. The graph executes:

```text
Stage 0 -> direct PP handoff -> Stage 1 -> sampling -> next token step
```

A fresh device trace assigned 185,900 of 188,702 kernels and found zero Stage1(A)/Stage0(B) overlaps across all 29 stable C4 decode pairs. The PP boundary itself averaged only `0.138 ms`; transport was direct with zero Meta fallback.

### Smaller decode microbatches do overlap, but lose Q8 efficiency

A two-plan prototype retained alternating recurrent/attention topologies, cached Meta child graphs, synchronized only Stage 0, and preserved direct PP transport. rocprof then proved seven positive overlaps with `17.742 ms` aggregate and `6.965 ms` maximum overlap.

It still lost:

- C4 two-row microbatches: `58.39 tok/s`, versus approximately `79 tok/s` unsplit.
- C8 four-row microbatches: `100.18 tok/s`, versus `110.85` unsplit hybrid and `139.77` TP4.
- All 16 fixed-seed C8 requests diverged from unsplit hybrid at generated token 1.

The reduced matrix row width costs more than overlap saves, and changing recurrent execution order fails deterministic equivalence.

### The serial TP2 body is the lower bound

For matched C4:

```text
TP2×PP2 cycle: 50.858 ms / four tokens
TP4 cycle:     42.269 ms / four tokens
```

Stable profiled hybrid composition was:

```text
Stage 0:             28.320 ms
PP boundary:          0.138 ms
Stage 1:             30.672 ms
Inter-dispatch gap:    3.231 ms
```

An intentionally optimistic projection removes all Stage-1-over-Stage-0 excess, the entire PP boundary, and even the complete hybrid inter-dispatch gap while leaving TP4 unchanged. It reaches only `86.60 tok/s`, still `8.49%` below TP4 at `94.63 tok/s`.

Thus transport, output imbalance, and host dispatch cannot close the measured gap. The two TP2 half-model executions remain serial for a single autoregressive batch. Changing that arithmetic topology converges back toward TP4.

## Experiments rejected

- `-ub 1/2`: generation fell to approximately `53/43 tok/s`.
- Disabling graph reuse: did not restore useful overlap.
- Two-entry context/Meta graph caches: created overlap but remained slower and changed output.
- Unified KV: `54.65 tok/s`; alternating recurrent topology remained.
- 28/36, 32/32, 36/28, 40/24, and 44/20 boundaries: no meaningful TG reversal. At C16, best 36/28 gained only `0.53%` over 32/32 and still lost TP4 by `14.95%`.
- Native gfx1030/GDN/Q8-cache body switches: hybrid TG `-2.78%`; TP4 TG `-1.09%` in the scoped component comparison.
- Full Qwen3.8 Q8 output/native profile: TP4 benefited more; hybrid remained `21.49%` behind C4 generation.
- Hidden-axis sharded output plus backend sampling: unsupported in the tested graph; failed before valid streamed results and was rejected.
- C16/C32 request batching: no TG crossover.

All rejected source prototypes were preserved as patches and reverted.

## Retained source changes

Only generic diagnostic tooling is retained on this follow-up branch:

- `scripts/analyze-parallel-critical-path.py`
- `GGML_META_TRACE` graph-rebuild telemetry
- scheduler fast-allocation/re-reserve telemetry under `GGML_SCHED_TRACE`

No failed runtime optimization is retained.

## Evidence

Root:

```text
/home/edwin/hybrid-tp-pp-evidence/tp2-pp2-beat-tp4-q8/
```

Principal artifacts:

- `iter1-server-pp512-tg128-c1c4/comparison.json`
- `iter2-device-profiles/hybrid/timeline-summary.json`
- `iter2-device-profiles/hybrid/critical-path.json`
- `iter3-meta-rebuild-instrument/meta-summary.json`
- `iter4-two-plan-device-profile/overlap-summary.json`
- `iter4-c8-ub4-probe/comparison.json`
- `iter4-c8-ub4-probe/output-comparison.json`
- `iter5-native-body-components/comparison.json`
- `iter6-c16-pp512-tg128/comparison.json`
- `iter6-c32-pp128-tg64/comparison.json`
- `iter6-c16-boundary-sweep/comparison.json`
- `iter7-backend-sampling-c4/comparison.json`
- `iter7-optimistic-lower-bound/analysis.json`

## Conclusion

The investigation met the diagnostic goal but not the requested performance goal. TP2×PP2 should not replace TP4 for Qwen3.8 Q8 generation. It remains useful for experimentation and can improve prompt capacity in some concurrent workloads, but those prompt gains do not imply a generation win.
