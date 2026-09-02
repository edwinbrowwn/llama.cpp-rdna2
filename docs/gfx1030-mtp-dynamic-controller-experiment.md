# gfx1030 MTP batch-depth controller

This candidate starts from smooth Q4 commit `1e3debc14`. It does not change target verification, proposal distributions, or the standard Q4 sidecar provider.

## Retained controller

The qualified gfx1030 TP4 vocabulary-sharded sidecar MTP profile automatically selects a batch-sticky depth schedule when the server is configured for one to four slots:

| Active neural MTP sequences | Draft depth |
|---:|---:|
| 1 | 4 |
| 2-3 | 3 |
| 4 | 2 |
| Other/unqualified | fixed configured depth |

K4V runs first. A sequence with an n-gram hit leaves the neural drafting set before active MTP sequences are counted, so K4V retains its bounded four-token chunk without paying an MTP step. The selected MTP cap is known before sidecar execution and therefore prevents unused serial draft work rather than truncating after computation.

The controller applies only to stochastic sidecar MTP. Temperature-zero and native draft implementations remain unchanged. An explicit setting always wins:

- unset: automatic `batch` mode on the exact qualified server profile; otherwise off.
- `batch`, `1`, or `on`: enable the validated buckets; unqualified active batch sizes retain fixed depth.
- `off` or `0`: retain fixed depth.
- `trace`: log the counterfactual batch bucket without changing depth.

## Correctness invariants

1. Target verification and rejection sampling are unchanged.
2. Sparse proposal-distribution rows remain aligned one-to-one with proposal tokens.
3. Shortening a stochastic proposal preserves the target distribution, though a fixed seed may follow a different valid trajectory.
4. Temperature-zero remains the deterministic output gate and bypasses dynamic capping.
5. The standard profile still verifies at most four K4V drafts. The explicit p1 capacity-five profile below verifies at most five; no 49-row target pass is restored.
6. Unsupported devices/output modes/stacks never receive automatic activation.

## Rejected approaches

- Per-cycle top-1 or sampled-proposal confidence produced sparse depth changes. Each change and usually the next width-four cycle took roughly 73-83 ms because the two-plan Meta cache and deferred width-five path churned. A conservative policy was 15.3% slower at p1.
- Acceptance-length feedback has the same per-cycle width-instability problem and is not part of the retained automatic path.
- Per-step device-to-host confidence synchronization was not implemented because the cheaper post-draft falsification already lost.
- Automatic graph suppression is no longer retained. The short fixed-work suite favored graph-off by 0.49%, but the user's long interactive workload was materially faster with graphs enabled. Graphs now follow the backend default; `GGML_CUDA_DISABLE_GRAPHS=1` remains an explicit tail-latency diagnostic.

## Measured schedule rationale

At p1, stable true depths 1/2/3/4 reached `59.793 / 75.721 / 71.337 / 80.468 tok/s`, so depth four remains unchanged. At p2, depth three beat depth two by 3.12%. At p4, depth two beat depth three by 42.3% (`130.491` versus `91.687 tok/s`) because the larger target geometry was particularly poor.

Repeated p2 true-early testing improved aggregate throughput from fixed-depth-four `86.327` to `118.017 tok/s`; repeated p4 improved `123.126` to `130.491 tok/s`. Tail latency improved in both cases. Automatic activation is capped at `--parallel 4`; larger servers retain fixed depth until separately qualified.

## Optional p1 bounded K4V width five

For the exact gfx1030 TP4 vocabulary-sharded sidecar stack `draft-mtp,ngram-map-k4v`, configuring `--parallel 1 --spec-draft-n-max 5` selects source-specific fixed request geometry:

- neural MTP remains depth four;
- K4V is capped at five even when its configured value width is 48;
- both five- and six-row target batches use deferred sidecar catch-up;
- HIP graphs remain enabled by default;
- a request-level `speculative.n_max` remains authoritative.

The mode is explicit because no-hit MTP-only requests were 1.13% slower in the graph-off ABBA. On the 512-token repetitive graph-off workload it improved `143.306 -> 152.127 tok/s` (`+6.16%`) and reduced 105 target cycles to 88. Those figures remain historical evidence for source-specific width, not a claim about the now-default graph-on policy; graph-on performance and tails must be reported separately. Temperature-zero output remained exact.

Set `LLAMA_SPEC_MTP_NEURAL_DEPTH=5` to diagnose the fully fixed depth-five mode. The standard and recommended general-workload setting remains `--spec-draft-n-max 4`.
