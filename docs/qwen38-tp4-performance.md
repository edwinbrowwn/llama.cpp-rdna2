# Qwen3.8 four-V620 TP4 performance profiles

This follow-up uses legacy four-rank TP4—not TP2—as the only performance bar. It does not claim that TP2×PP2 beats TP4. The work remains isolated on `perf/hybrid-beat-tp4` and is not merged to `master`.

## Profiles

### Exact Q8 profile

The exact profile keeps the qualified Qwen3.8 Q8 target and enables vocabulary-axis output-head sharding:

```bash
GGML_TP_SHARDED_OUTPUT=1
GGML_TP_VOCAB_SHARDED_OUTPUT=1
```

Each TP rank computes a disjoint vocabulary range. The Meta buffer gathers those ranges in vocabulary order for the normal host sampler. There is no output-head all-reduce and no redundant full-vocabulary matmul. Transformer arithmetic is unchanged.

A deterministic `4 × 248,320` full-logit comparison was byte-identical to the mirrored output head across all 993,280 F32 values (`max_abs=0`, `RMSE=0`) with identical argmax tokens `271, 6161, 271, 271`.

Exact-HEAD ABBA `llama-bench`, 14 samples per arm:

| Q8 TP4 configuration | PP512 | TG128 |
|---|---:|---:|
| mirrored output baseline | 1428.224 tok/s | 29.326 tok/s |
| vocabulary-sharded output | **1448.305 tok/s** | **31.842 tok/s** |
| change | **+1.406%** | **+8.582%** |

PP is sensitive to process order and GPU state, so the PP improvement is intentionally described as small. TG is stable and materially faster.

Evidence:

- `/home/edwin/hybrid-tp-pp-evidence/beat-tp4/q8-vocab-abba-pp512-tg128-r7-451af1939/`
- `/home/edwin/hybrid-tp-pp-evidence/beat-tp4-candidates/vocab-shard-numerical-rccl-dirty-a3faa325c/`

### Maximum-throughput profile

The maximum profile uses the retained auto20 Q4S8 model and combines:

```bash
GGML_HIP_GFX1030_NATIVE=1
GGML_HIP_GFX1030_GDN_SIBLING_FUSION=1
GGML_HIP_GFX1030_Q8_CACHE=1
GGML_TP_SHARDED_OUTPUT=1
GGML_TP_VOCAB_SHARDED_OUTPUT=1
```

Exact-HEAD ABBA `llama-bench`, 14 samples per arm, against the original Q8 TP4 bar:

| TP4 configuration | PP512 | TG128 |
|---|---:|---:|
| original Q8 baseline | 1397.612 tok/s | 29.279 tok/s |
| auto20 Q4S8 maximum profile | **1442.849 tok/s** | **39.275 tok/s** |
| change | **+3.237%** | **+34.140%** |

This is not an equal-quantization comparison. The auto20 artifact's retained KLD against BF16 is `0.025422`, versus `0.000937` for Q8. Native GDN changes can also change greedy output when logits are nearly tied. Use the exact Q8 profile when bit-identical full logits are required.

Evidence:

- `/home/edwin/hybrid-tp-pp-evidence/beat-tp4/max-q4s8-vs-q8-abba-pp512-tg128-r7-451af1939/`
- `/home/edwin/hybrid-tp-pp-evidence/beat-tp4/q4s8-q8only-fusion-greedy128-451af1939/`

## Usage

```bash
# Repeated PP512/TG128 benchmark with the exact Q8 profile
./scripts/run-qwen38-tp4-fast.sh exact bench -p 512 -n 128 -r 5 -o json

# Maximum-throughput server profile
./scripts/run-qwen38-tp4-fast.sh max server -c 8192 --parallel 4
```

Both profiles retain tensor split and explicit four-device ordering. The wrapper sets the qualified ROCm environment, disables HIP graphs, selects RCCL, and leaves all optimization switches explicit.

## Rejected paths

The following did not beat TP4 and are not recommended:

- TP1×PP4: about 416 PP tok/s and 16.1 TG tok/s at concurrency 1 in the initial PP512/TG128 probe.
- Hidden-dimension output sharding: increased all-reduce work and reduced high-concurrency throughput.
- HIP graphs on this graph: reduced throughput.
- Internal all-reduce instead of RCCL: reduced throughput.
- Moving the MTP draft among stage-0/stage-1 or one/two GPUs did not close the TP4 gap.
- Extending sibling fusion to Q4_0 rows reached 1514.8 PP and 40.8 TG tok/s in one run, but changed the first greedy token and produced larger logit differences (`max_abs=0.535`, minimum correlation `0.99852`). The uncommitted prototype was rejected and reverted.

## Build and correctness gates

- Focused suite: arg parser, Qwen35MoE MMQ config, tensor split, Meta split, and parallel topology: 5/5 passed.
- Scoped Gated DeltaNet backend tests: stock 36/36 and native 36/36 on ROCm0.
- Required build: Release HIP/gfx1030, shared libraries, `GGML_HIP_RCCL=ON`, `GGML_CUDA_NCCL=ON`, HIP graphs compiled but disabled at runtime.
- Current optimization commit: `451af1939` (plus the generic benchmark-rank fix `fd3c171cf`).
