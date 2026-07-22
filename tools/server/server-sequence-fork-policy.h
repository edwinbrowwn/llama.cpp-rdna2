#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

#include "ggml.h"

namespace server_sequence_fork_policy {

constexpr size_t restored_suffix_ratio_denominator = 12;

inline bool cache_types_support_vector(ggml_type type_k, ggml_type type_v) {
    if (type_k != type_v) {
        return false;
    }
    switch (type_k) {
        case GGML_TYPE_F16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_BF16:
            return true;
        default:
            return false;
    }
}

inline bool should_use_vector(
        bool   is_amd,
        bool   vector_supported,
        size_t prompt_tokens,
        size_t cached_tokens) {
    if (!is_amd || !vector_supported || prompt_tokens == 0 || cached_tokens == 0 || cached_tokens > prompt_tokens) {
        return false;
    }

    const size_t suffix_tokens = prompt_tokens - cached_tokens;
    return suffix_tokens <= prompt_tokens/restored_suffix_ratio_denominator;
}

class restored_suffix_state {
public:
    bool start(size_t tokens) {
        if (tokens == 0 || tokens > std::numeric_limits<uint32_t>::max()) {
            clear_all();
            return false;
        }
        remaining_ = (uint32_t) tokens;
        force_vector_ = true;
        return true;
    }

    // Clear task-local suffix accounting while preserving the requirement that
    // any subsequent use of this restored sequence remains on vector FA.
    void clear_prompt() {
        remaining_ = 0;
    }

    // Clear both accounting and restored-state provenance after the underlying
    // sequence memory has been atomically discarded and rebuilt.
    void clear_all() {
        remaining_ = 0;
        force_vector_ = false;
    }

    void mark_restored() {
        force_vector_ = true;
    }

    bool active() const {
        return remaining_ > 0;
    }

    bool force_vector() const {
        return force_vector_;
    }

    uint32_t remaining() const {
        return remaining_;
    }

    bool consume(size_t tokens) {
        if (tokens > remaining_) {
            return false;
        }
        remaining_ -= (uint32_t) tokens;
        return true;
    }

private:
    uint32_t remaining_ = 0;
    bool force_vector_ = false;
};

template <typename IsVector>
size_t homogeneous_prefix(size_t max_tokens, IsVector && is_vector) {
    if (max_tokens == 0) {
        return 0;
    }

    const bool first = is_vector(0);
    size_t i = 1;
    for (; i < max_tokens; ++i) {
        if (is_vector(i) != first) {
            break;
        }
    }
    return i;
}

} // namespace server_sequence_fork_policy