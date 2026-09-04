#pragma once

#include "speculative.h"

#include <algorithm>
#include <cstdint>

// A stacked sidecar n-gram proposal is capped at the configured neural width
// by default. A per-request speculative.n_max override remains authoritative
// and bypasses this internal anti-stutter cap.
struct common_speculative_sidecar_cap_config {
    int width = 0;
};

// Dense-MTP deferred catch-up must accept every target row when a stacked K4V
// proposal exceeds the fixed neural width. Content selection widens only K4V;
// the reserved target envelope still has N+1 verification rows.
inline bool common_speculative_sidecar_mtp_deferred_width_eligible(
        int base_nmax, int content_extra, int verification_rows) {
    if (base_nmax == 4) {
        return verification_rows >= 5 &&
                verification_rows <= base_nmax + std::max(0, content_extra) + 1;
    }
    return base_nmax == 5 && (verification_rows == 5 || verification_rows == 6);
}

// A request/capacity envelope may exceed the neural provider's configured
// width for stacked K4V verification. Keep neural generation fixed regardless.
inline int common_speculative_neural_draft_limit(
        int configured_nmax, int request_envelope) {
    if (configured_nmax <= 0) {
        return 0;
    }
    if (request_envelope == 0) {
        return 0;
    }
    return request_envelope > 0
            ? std::min(configured_nmax, request_envelope)
            : configured_nmax;
}

inline int common_speculative_proposal_limit(
        const common_speculative_draft_params & dp, bool is_k4v) {
    return is_k4v && dp.n_max_ngram > 0 ? dp.n_max_ngram : dp.n_max;
}

inline bool common_speculative_sidecar_cap_request_enabled(
        const common_speculative_sidecar_cap_config & config,
        const common_speculative_draft_params & dp) {
    return config.width > 0 && !dp.n_max_user_override;
}

inline int common_speculative_sidecar_cap_limit(
        const common_speculative_sidecar_cap_config & config,
        const common_speculative_draft_params & dp) {
    int limit = config.width;
    if (dp.n_max > 0) {
        limit = std::min(limit, dp.n_max);
    }
    if (dp.n_max_ngram > 0) {
        limit = std::min(limit, dp.n_max_ngram);
    }
    return std::max(0, limit);
}

inline void common_speculative_sidecar_cap_trim(
        const common_speculative_sidecar_cap_config & config,
        const common_speculative_draft_params & dp,
        llama_tokens & result) {
    if (result.empty() || !common_speculative_sidecar_cap_request_enabled(config, dp)) {
        return;
    }

    const int limit = common_speculative_sidecar_cap_limit(config, dp);
    if (limit > 0 && (int) result.size() > limit) {
        result.resize((size_t) limit);
    }
}
