#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// The content-aware controller is deliberately sidecar-only in this POC.
// These helpers are also used by buffer-sizing and sidecar-cap policy code.
bool common_speculative_content_env_enabled();
int common_speculative_content_env_max_boost();

enum spec_content_boost_level : uint8_t {
    SPEC_BOOST_0 = 0,
    SPEC_BOOST_1 = 1,
    SPEC_BOOST_2 = 2,
    SPEC_BOOST_3 = 3,
};

enum spec_token_flag : uint16_t {
    STF_NONE          = 0,
    STF_ALPHA         = 1u << 0,
    STF_DIGIT         = 1u << 1,
    STF_WHITESPACE    = 1u << 2,
    STF_NEWLINE       = 1u << 3,
    STF_QUOTE         = 1u << 4,
    STF_COLON         = 1u << 5,
    STF_COMMA         = 1u << 6,
    STF_BRACE         = 1u << 7,
    STF_BRACKET       = 1u << 8,
    STF_PAREN         = 1u << 9,
    STF_OPERATOR      = 1u << 10,
    STF_CODE_HINT     = 1u << 11,
    STF_MATH_HINT     = 1u << 12,
    STF_FENCE         = 1u << 13,
    STF_COMMENT_HINT  = 1u << 14,
    STF_STRONG_ANCHOR = 1u << 15,
};

static constexpr uint8_t SPEC_CONTENT_WINDOW_MAX = 16;

struct spec_content_roll_entry {
    llama_token token = -1;
    uint16_t flags = STF_NONE;
};

// Fixed-size so a provisional copy is cheap and cannot allocate in the
// generation hot path.  The state is never used to alter target verification.
struct common_speculative_content_state {
    uint32_t token_count = 0;
    int16_t structure_score = 0;

    // These are bounded local-structure activity counters, not a parser stack.
    uint8_t brace_depth = 0;
    uint8_t bracket_depth = 0;
    uint8_t paren_depth = 0;

    uint8_t quote_recent = 0;
    uint8_t colon_recent = 0;
    uint8_t newline_recent = 0;
    uint8_t operator_recent = 0;
    uint8_t repetition_score = 0;

    spec_content_boost_level current_level = SPEC_BOOST_0;
    spec_content_boost_level candidate_level = SPEC_BOOST_0;
    uint8_t candidate_age = 0;
    uint8_t transition_cooldown = 0;

    uint8_t history_size = 0;
    uint8_t history_pos = 0;
    spec_content_roll_entry history[SPEC_CONTENT_WINDOW_MAX] = {};

    // Passive cycle telemetry.  These fields do not feed back into selection.
    uint8_t level_before = 0;
    uint8_t level_at_base = 0;
    uint8_t level_final = 0;
    uint8_t level_after = 0;
    uint8_t boost_used = 0;
    int32_t base_nmax = 0;
    int32_t final_nmax = 0;
    uint32_t drafted = 0;
    uint32_t accepted = 0;
    bool cycle_valid = false;
};

struct common_speculative_content_config {
    bool enabled = false;
    // Sidecar providers draft a whole block in one call.  Provisional scoring
    // is therefore telemetry-only until a continuation API can extend safely.
    bool provisional = false;
    bool trace = false;
    bool adaptive = false;
    int max_boost = 3;
    int window = 12;
    int hysteresis = 2;
};

struct common_speculative_content_stats {
    uint64_t cycles = 0;
    uint64_t drafted = 0;
    uint64_t accepted = 0;
};

class common_speculative_content {
public:
    void init(const llama_vocab * vocab, uint32_t n_seq, bool sidecar_only,
              int base_nmax);

    bool enabled() const { return config_.enabled; }
    bool provisional() const { return config_.provisional; }
    bool trace() const { return config_.trace; }
    int max_boost() const { return config_.max_boost; }
    int base_nmax() const { return base_nmax_; }
    const common_speculative_content_config & config() const { return config_; }

    common_speculative_content_state * state(uint32_t seq_id);
    const common_speculative_content_state * state(uint32_t seq_id) const;

    void begin(uint32_t seq_id, const llama_token * tokens, size_t n_tokens);
    uint8_t prepare(uint32_t seq_id, const llama_token * tokens, size_t n_tokens,
                    llama_token id_last);

    void push(common_speculative_content_state & state, llama_token token) const;
    uint8_t level(const common_speculative_content_state & state) const;

    int max_limit(int base_nmax, int context_limit, bool user_override) const;
    int selected_limit(int base_nmax, int context_limit, bool user_override,
                       uint8_t level_before) const;

    void observe_draft(uint32_t seq_id, const llama_token * tokens, size_t n_tokens,
                       int base_nmax, int selected_nmax, int hard_nmax);
    void accept(uint32_t seq_id, const llama_token * tokens, size_t n_tokens,
                uint16_t n_committed);
    void accept(uint32_t seq_id, const llama_token * tokens, size_t n_tokens,
                uint16_t n_committed, uint16_t n_accepted_draft);
    void reset(uint32_t seq_id);

    void print_stats() const;

private:
    friend struct common_speculative_content_test_access;

    uint16_t token_flags(llama_token token) const;
    uint8_t score_level(int score) const;
    void extend_window(common_speculative_content_state & state,
                       const llama_token * tokens, size_t n_tokens) const;
    void update_level(common_speculative_content_state & state,
                      uint8_t desired, bool strong_anchor, bool repeated) const;

    common_speculative_content_config config_;
    int base_nmax_ = 0;
    std::vector<uint16_t> token_flags_;
    std::vector<common_speculative_content_state> states_;
    common_speculative_content_stats stats_[4] = {};
};
