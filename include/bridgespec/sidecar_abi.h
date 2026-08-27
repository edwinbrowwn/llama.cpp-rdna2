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

// Qwen3.8-27B MTP sidecar ABI (release ABI 2).
BRIDGESPEC_API int spec_hip_release_abi(void);
BRIDGESPEC_API int spec_hip_check(int32_t n_embd, int32_t head_rows);
BRIDGESPEC_API int spec_hip_state_size(void);
BRIDGESPEC_API int spec_hip_get_state(void * data, int size);
BRIDGESPEC_API int spec_hip_set_state(const void * data, int size);
BRIDGESPEC_API int spec_hip_reset_state(void);
BRIDGESPEC_API int spec_hip_truncate_state(int32_t pos_max);
BRIDGESPEC_API int spec_hip_rebase_state(int32_t pos_min, int32_t pos_max, int32_t delta);
BRIDGESPEC_API int spec_hip_init(const char * weights_dir, const char * ids_path);
BRIDGESPEC_API int spec_hip_catchup(
        const int32_t * tokens,
        const int32_t * positions,
        const float * hidden_rows,
        int count);
BRIDGESPEC_API int spec_hip_draft(
        int32_t last_token,
        int32_t past_tokens,
        const float * hidden,
        int max_draft,
        int32_t * output_ids);

// Qwen3.8-27B DFlash sidecar ABI (release ABI 3).
BRIDGESPEC_API int spec_dflash_release_abi(void);
BRIDGESPEC_API int spec_dflash_check(int32_t encoded_width, int32_t block_size);
BRIDGESPEC_API int spec_dflash_state_size(void);
BRIDGESPEC_API int spec_dflash_get_state(void * data, int size);
BRIDGESPEC_API int spec_dflash_set_state(const void * data, int size);
BRIDGESPEC_API int spec_dflash_reset_state(void);
BRIDGESPEC_API int spec_dflash_truncate_state(int32_t pos_max);
BRIDGESPEC_API int spec_dflash_rebase_state(int32_t pos_min, int32_t pos_max, int32_t delta);
BRIDGESPEC_API int spec_dflash_init(const char * artifact_directory);
BRIDGESPEC_API int spec_dflash_chunk(
        const int32_t * positions,
        const float * target_features,
        int count);
BRIDGESPEC_API int spec_dflash_draft(
        int32_t last_token,
        int32_t past_tokens,
        int32_t * output_ids);

#ifdef __cplusplus
}
#endif
