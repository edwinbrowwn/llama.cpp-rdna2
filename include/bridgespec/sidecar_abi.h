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

// Qwen3.8-27B MTP sidecar ABI (release ABI 1).
BRIDGESPEC_API int spec_hip_release_abi(void);
BRIDGESPEC_API int spec_hip_check(int32_t n_embd, int32_t head_rows);
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

// Qwen3.8-27B DFlash sidecar ABI (release ABI 2).
BRIDGESPEC_API int spec_dflash_release_abi(void);
BRIDGESPEC_API int spec_dflash_check(int32_t encoded_width, int32_t block_size);
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
