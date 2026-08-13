#pragma once

#include <algorithm>
#include <cstdint>

struct server_speculative_draft_limit_input {
    int32_t available;
    int32_t request_max;
    bool reasoning_pause;
    bool reasoning_active;
};

inline int32_t server_speculative_draft_limit(const server_speculative_draft_limit_input & input) {
    int32_t result = input.available;
    if (input.request_max >= 0) {
        result = std::min(result, input.request_max);
    }
    if (input.reasoning_pause && input.reasoning_active) {
        return 0;
    }
    return result;
}
