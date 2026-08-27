// SPDX-License-Identifier: MIT
#pragma once

#include <stdint.h>

#if defined(_WIN32)
#  if defined(BRIDGESPEC_BUILDING_DLL)
#    define BRIDGESPEC_API __declspec(dllexport)
#  else
#    define BRIDGESPEC_API __declspec(dllimport)
#  endif
#else
#  define BRIDGESPEC_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Sidecar state snapshots contain only a cheap logical cursor and an epoch.
// The large device KV cache remains resident; callers must use the same loaded
// sidecar instance when restoring a snapshot.
#define BRIDGESPEC_STATE_MAGIC   UINT32_C(0x42535354) // "BSST"
#define BRIDGESPEC_STATE_VERSION UINT16_C(1)

enum bridgespec_state_kind {
    BRIDGESPEC_STATE_KIND_MTP    = 1,
    BRIDGESPEC_STATE_KIND_DFLASH = 2,
};

struct bridgespec_sidecar_state {
    uint32_t magic;
    uint16_t version;
    uint16_t kind;
    int32_t  pos_min;
    int32_t  pos_max; // exclusive
    uint64_t epoch;
};

#ifdef __cplusplus
static_assert(sizeof(bridgespec_sidecar_state) == 24,
              "BridgeSpec state ABI must remain a fixed 24-byte record");
#endif

// The sidecars use a small top-k proposal distribution for stochastic
// drafting. The distribution is returned in row-major [max_draft][top_k]
// buffers; only the first returned-token-count rows are valid.
#define BRIDGESPEC_MTP_DRAFT_TOP_K    32
#define BRIDGESPEC_DFLASH_DRAFT_TOP_K 16

// Qwen3.8-27B MTP sidecar ABI (release ABI 4).
// State and KV operations are sequence-scoped. catchup writes only pending
// target rows; commit_state is the only operation that makes rows persistent.
// hidden_device, when non-null, is the already-selected accepted hidden row.
BRIDGESPEC_API int spec_hip_release_abi(void);
BRIDGESPEC_API int spec_hip_check(int32_t n_embd, int32_t head_rows, int32_t n_seq);
BRIDGESPEC_API int spec_hip_state_size(void);
BRIDGESPEC_API int spec_hip_get_state(int32_t seq_id, void * data, int size);
BRIDGESPEC_API int spec_hip_set_state(int32_t seq_id, const void * data, int size);
BRIDGESPEC_API int spec_hip_reset_state(int32_t seq_id);
BRIDGESPEC_API int spec_hip_truncate_state(int32_t seq_id, int32_t pos_max);
BRIDGESPEC_API int spec_hip_commit_state(int32_t seq_id, int32_t pos_max, const float * hidden_device);
BRIDGESPEC_API int spec_hip_rebase_state(int32_t seq_id, int32_t pos_min, int32_t pos_max, int32_t delta);
// stream is borrowed from the target HIP backend and must remain valid for
// the sidecar lifetime. Passing null restores the sidecar-owned stream.
BRIDGESPEC_API int spec_hip_attach_target_stream(void * stream, int32_t device);
BRIDGESPEC_API int spec_hip_init(const char * weights_dir, const char * ids_path, int32_t n_seq);
BRIDGESPEC_API int spec_hip_catchup(
        int32_t seq_id,
        const int32_t * tokens,
        const int32_t * positions,
        const float * hidden_rows,
        int count);
BRIDGESPEC_API int spec_hip_catchup_device(
        int32_t seq_id,
        const int32_t * tokens,
        const int32_t * positions,
        const float * hidden_rows_device,
        int count);
BRIDGESPEC_API int spec_hip_draft(
        int32_t seq_id,
        int32_t last_token,
        int32_t past_tokens,
        const float * hidden,
        int max_draft,
        int32_t * output_ids);
BRIDGESPEC_API int spec_hip_draft_device(
        int32_t seq_id,
        int32_t last_token,
        int32_t past_tokens,
        int max_draft,
        int32_t * output_ids);
// Stochastic variants use the supplied keyed RNG stream. They return only
// selected IDs and compact q distributions; MTP samples on-device while
// DFlash reuses its compact selector readback. The target sampler remains the
// owner of verifier RNG/rejection semantics.
BRIDGESPEC_API int spec_hip_stochastic_top_k(void);
BRIDGESPEC_API int spec_hip_draft_stochastic(
        int32_t seq_id,
        int32_t last_token,
        int32_t past_tokens,
        const float * hidden,
        float temperature,
        float p_min,
        uint64_t rng_key,
        int max_draft,
        int32_t * output_ids,
        int32_t * dist_ids,
        float * dist_probs);
BRIDGESPEC_API int spec_hip_draft_stochastic_device(
        int32_t seq_id,
        int32_t last_token,
        int32_t past_tokens,
        float temperature,
        float p_min,
        uint64_t rng_key,
        int max_draft,
        int32_t * output_ids,
        int32_t * dist_ids,
        float * dist_probs);

// Qwen3.8-27B DFlash sidecar ABI (release ABI 5). Target chunks are staged
// until commit_state; device-layer input pointers are borrowed for the call.
BRIDGESPEC_API int spec_dflash_release_abi(void);
BRIDGESPEC_API int spec_dflash_check(int32_t encoded_width, int32_t block_size, int32_t n_seq);
BRIDGESPEC_API int spec_dflash_state_size(void);
BRIDGESPEC_API int spec_dflash_get_state(int32_t seq_id, void * data, int size);
BRIDGESPEC_API int spec_dflash_set_state(int32_t seq_id, const void * data, int size);
BRIDGESPEC_API int spec_dflash_reset_state(int32_t seq_id);
BRIDGESPEC_API int spec_dflash_truncate_state(int32_t seq_id, int32_t pos_max);
BRIDGESPEC_API int spec_dflash_commit_state(int32_t seq_id, int32_t pos_max);
BRIDGESPEC_API int spec_dflash_rebase_state(int32_t seq_id, int32_t pos_min, int32_t pos_max, int32_t delta);
BRIDGESPEC_API int spec_dflash_attach_target_stream(void * stream, int32_t device);
BRIDGESPEC_API int spec_dflash_init(const char * artifact_directory, int32_t n_seq);
BRIDGESPEC_API int spec_dflash_chunk(
        int32_t seq_id,
        const int32_t * positions,
        const float * target_features,
        int count);
BRIDGESPEC_API int spec_dflash_chunk_device(
        int32_t seq_id,
        const int32_t * positions,
        const void * const * target_layer_features_device,
        int n_layers,
        int layer_width,
        int count);
BRIDGESPEC_API int spec_dflash_draft(
        int32_t seq_id,
        int32_t last_token,
        int32_t past_tokens,
        int32_t * output_ids);
BRIDGESPEC_API int spec_dflash_stochastic_top_k(void);
BRIDGESPEC_API int spec_dflash_draft_stochastic(
        int32_t seq_id,
        int32_t last_token,
        int32_t past_tokens,
        float temperature,
        float p_min,
        uint64_t rng_key,
        int max_draft,
        int32_t * output_ids,
        int32_t * dist_ids,
        float * dist_probs);

#ifdef __cplusplus
}

#if defined(__HIPCC__)
#  define BRIDGESPEC_HD __host__ __device__
#else
#  define BRIDGESPEC_HD
#endif

// Counter-based proposal RNG. It is deliberately independent of the target
// sampler's rejection RNG and needs no mutable state in the 24-byte snapshot.
static inline BRIDGESPEC_HD uint64_t bridgespec_stochastic_mix64(uint64_t x) {
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

static inline BRIDGESPEC_HD double bridgespec_stochastic_uniform(uint64_t key, uint32_t step) {
    const uint64_t z = bridgespec_stochastic_mix64(
            key + UINT64_C(0xd1b54a32d192ed03) * (uint64_t) (step + 1));
    return (double) (z >> 11) * (1.0 / 9007199254740992.0);
}

#undef BRIDGESPEC_HD
#endif
