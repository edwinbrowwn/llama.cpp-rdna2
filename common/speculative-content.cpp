#include "speculative-content.h"

#include "common.h"
#include "log.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>

namespace {

static bool env_switch_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr &&
        (std::strcmp(value, "1") == 0 || std::strcmp(value, "on") == 0 ||
         std::strcmp(value, "true") == 0);
}

static int env_int(const char * name, int fallback, int low, int high) {
    const char * value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }

    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0') {
        return fallback;
    }
    return (int) std::clamp(parsed, (long) low, (long) high);
}

static void warn_env_adjustment(const char * name, int effective,
        int low, int high) {
    const char * value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return;
    }

    char * end = nullptr;
    const long requested = std::strtol(value, &end, 10);
    if (end == value || *end != '\0') {
        LOG_WRN("spec content %s='%s' is invalid; using %d\n",
                name, value, effective);
        return;
    }
    if (requested < low || requested > high) {
        LOG_WRN("spec content %s=%ld is outside [%d,%d]; clamped to %d\n",
                name, requested, low, high, effective);
    }
}

static bool is_word_char(unsigned char c) {
    return std::isalnum(c) != 0 || c == '_';
}

static unsigned char ascii_lower(unsigned char c) {
    return c >= 'A' && c <= 'Z' ? (unsigned char) (c + ('a' - 'A')) : c;
}

static bool contains_word(const char * text, const char * word) {
    if (text == nullptr || word == nullptr || *word == '\0') {
        return false;
    }

    const size_t text_len = std::strlen(text);
    const size_t word_len = std::strlen(word);
    if (word_len > text_len) {
        return false;
    }

    for (size_t i = 0; i + word_len <= text_len; ++i) {
        if (i > 0 && is_word_char((unsigned char) text[i - 1])) {
            continue;
        }
        if (i + word_len < text_len && is_word_char((unsigned char) text[i + word_len])) {
            continue;
        }

        bool match = true;
        for (size_t j = 0; j < word_len; ++j) {
            if (ascii_lower((unsigned char) text[i + j]) !=
                    ascii_lower((unsigned char) word[j])) {
                match = false;
                break;
            }
        }
        if (match) {
            return true;
        }
    }
    return false;
}

static bool contains_text(const char * text, const char * needle) {
    return text != nullptr && needle != nullptr && std::strstr(text, needle) != nullptr;
}

static bool content_inline_backtick_parity(const char * text) {
    if (text == nullptr || std::strstr(text, "```") != nullptr) {
        return false;
    }
    bool odd = false;
    for (const char * p = text; *p != 0; ) {
        if (*p != '`') {
            ++p;
            continue;
        }
        odd = !odd;
        while (*p == '`') ++p;
    }
    return odd;
}

static bool starts_with_hash_comment(const char * text) {
    if (text == nullptr) {
        return false;
    }
    const unsigned char * p = (const unsigned char *) text;
    while (*p == ' ' || *p == '\t') {
        ++p;
    }
    return *p == '#';
}

static uint16_t content_piece_flags(const char * text) {
    uint16_t flags = STF_NONE;
    if (text == nullptr) {
        return flags;
    }

    for (const unsigned char * p = (const unsigned char *) text; *p != 0; ++p) {
        const unsigned char c = *p;
        if (std::isalpha(c) != 0) flags |= STF_ALPHA;
        if (std::isdigit(c) != 0) flags |= STF_DIGIT;
        if (c == ' ' || c == '\t' || c == '\f' || c == '\v') flags |= STF_WHITESPACE;
        if (c == '\n' || c == '\r') flags |= STF_NEWLINE;
        if (c == '\'' || c == '"' || c == '`') flags |= STF_QUOTE;
        if (c == ':') flags |= STF_COLON;
        if (c == ',') flags |= STF_COMMA;
        if (c == '{' || c == '}') flags |= STF_BRACE;
        if (c == '[' || c == ']') flags |= STF_BRACKET;
        if (c == '(' || c == ')') flags |= STF_PAREN;
        if (std::strchr("=+-*/^<>&;", c) != nullptr) flags |= STF_OPERATOR;
    }

    static const char * const code_words[] = {
        "return", "def", "function", "const", "let", "import", "struct",
        "enum", "switch",
        nullptr,
    };
    for (const char * const * word = code_words; *word != nullptr; ++word) {
        if (contains_word(text, *word)) {
            flags |= STF_CODE_HINT;
            break;
        }
    }

    if (contains_text(text, "==") || contains_text(text, "!=") ||
            contains_text(text, "->") || contains_text(text, "=>") ||
            contains_text(text, "::") || contains_text(text, "&&") ||
            contains_text(text, "||")) {
        flags |= STF_CODE_HINT | STF_OPERATOR;
    }

    // Markdown emphasis and tokenizer-split hyphenated words are common in
    // prose reasoning. Do not let their punctuation masquerade as source
    // operators; surrounding source syntax remains independently sufficient.
    if (contains_text(text, "**") && !contains_text(text, "**=")) {
        flags &= (uint16_t) ~STF_OPERATOR;
    }
    if ((flags & STF_ALPHA) != 0 && (flags & STF_OPERATOR) != 0) {
        bool only_hyphen_operators = true;
        for (const unsigned char * p = (const unsigned char *) text; *p != 0; ++p) {
            if (std::strchr("=+-*/^<>&;", *p) != nullptr && *p != '-') {
                only_hyphen_operators = false;
                break;
            }
        }
        if (only_hyphen_operators && !contains_text(text, "--")) {
            flags &= (uint16_t) ~STF_OPERATOR;
        }
    }

    if (contains_text(text, "---")) {
        bool markdown_rule = true;
        for (const unsigned char * p = (const unsigned char *) text; *p != 0; ++p) {
            if (*p < 0x80 && *p != '-' && *p != ':' && *p != '|' &&
                    *p != ' ' && *p != '\t' && *p != '\r' && *p != '\n') {
                markdown_rule = false;
                break;
            }
        }
        if (markdown_rule) {
            flags &= (uint16_t) ~(STF_OPERATOR | STF_CODE_HINT);
        }
    }

    if (contains_text(text, "\\frac") || contains_text(text, "\\sum") ||
            contains_text(text, "\\sqrt") || contains_text(text, "\\int")) {
        flags |= STF_MATH_HINT;
    }

    if (contains_text(text, "```")) {
        flags |= STF_FENCE;
    }
    const bool url_scheme = contains_text(text, "://");
    if ((!url_scheme && contains_text(text, "//")) || contains_text(text, "/*") ||
            contains_text(text, "*/") || starts_with_hash_comment(text)) {
        flags |= STF_COMMENT_HINT;
    }
    if (url_scheme) {
        flags &= (uint16_t) ~STF_OPERATOR;
    } else if (text[0] == '/' && text[1] != 0) {
        bool path_piece = true;
        for (const unsigned char * p = (const unsigned char *) text + 1; *p != 0; ++p) {
            if (std::isalnum(*p) == 0 && *p != '_' && *p != '-' && *p != '.') {
                path_piece = false;
                break;
            }
        }
        if (path_piece) flags &= (uint16_t) ~STF_OPERATOR;
    }

    if ((flags & STF_FENCE) != 0) {
        flags |= STF_STRONG_ANCHOR;
    }
    return flags;
}

static const spec_content_roll_entry * history_back(
        const common_speculative_content_state & state, size_t back,
        uint8_t capacity) {
    if (capacity == 0 || capacity > SPEC_CONTENT_WINDOW_MAX ||
            back >= state.history_size) {
        return nullptr;
    }

    int newest;
    if (state.history_size < capacity) {
        newest = (int) state.history_size - 1;
    } else {
        newest = ((int) state.history_pos + capacity - 1) % capacity;
    }
    const int index = (newest - (int) back + capacity) % capacity;
    return &state.history[index];
}

static int clamp_score(int score) {
    return std::clamp(score, -16, 24);
}

} // namespace

uint16_t common_speculative_content_piece_flags(const char * text) {
    return content_piece_flags(text);
}

bool common_speculative_content_inline_backtick_parity(const char * text) {
    return content_inline_backtick_parity(text);
}

bool common_speculative_content_token_attr_eligible(llama_token_attr attr) {
    constexpr uint32_t excluded = LLAMA_TOKEN_ATTR_UNKNOWN |
            LLAMA_TOKEN_ATTR_UNUSED | LLAMA_TOKEN_ATTR_CONTROL |
            LLAMA_TOKEN_ATTR_USER_DEFINED | LLAMA_TOKEN_ATTR_BYTE;
    return ((uint32_t) attr & LLAMA_TOKEN_ATTR_NORMAL) != 0 &&
            ((uint32_t) attr & excluded) == 0;
}

bool common_speculative_content_stack_eligible(
        const common_params_speculative & params) {
    int n_mtp = 0;
    int n_k4v = 0;
    for (common_speculative_type type : params.types) {
        switch (type) {
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:     ++n_mtp; break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: ++n_k4v; break;
            case COMMON_SPECULATIVE_TYPE_NONE:          break;
            default: return false;
        }
    }
    // Without K4V there is nothing to widen: content selection must never
    // change the sole neural provider's generation width. DFlash remains fixed
    // because its matched K4V5 verification screen was negative.
    return n_mtp == 1 && n_k4v == 1 && params.draft.n_max == 4;
}

bool common_speculative_content_candidate_eligible(
        const common_params_speculative & params) {
    return params.draft.content_verification_eligible &&
            params.draft.sidecar_candidate_ready && !params.has_synth() &&
            common_speculative_content_stack_eligible(params);
}

bool common_speculative_content_runtime_eligible(
        const common_params_speculative & params) {
    return common_speculative_content_candidate_eligible(params) &&
            params.draft.sidecar_only &&
            params.draft.sidecar_type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP;
}

bool common_speculative_content_env_enabled() {
    // Sidecar activation remains the master gate and intentionally requires
    // the exact value "1". Once active, stacked MTP verification defaults on;
    // SPEC_CONTENT_BOOST is a backward-compatible explicit enable/disable.
    const char * sidecar = std::getenv("SPEC_SIDECAR");
    if (sidecar == nullptr || std::strcmp(sidecar, "1") != 0) {
        return false;
    }
    const char * content = std::getenv("SPEC_CONTENT_BOOST");
    if (content != nullptr && !env_switch_enabled("SPEC_CONTENT_BOOST")) {
        return false;
    }
    return common_speculative_content_env_max_boost() > 0;
}

int common_speculative_content_env_max_boost() {
    return env_int("SPEC_CONTENT_MAX_BOOST", 1, 0, 3);
}

void common_speculative_content::init(
        const llama_vocab * vocab, uint32_t n_seq, bool eligible, int base_nmax) {
    config_ = {};
    config_.enabled = eligible && common_speculative_content_env_enabled();
    config_.provisional = env_switch_enabled("SPEC_CONTENT_PROVISIONAL");
    config_.trace = env_switch_enabled("SPEC_CONTENT_TRACE");
    config_.adaptive = env_switch_enabled("SPEC_CONTENT_ADAPT");
    config_.max_boost = common_speculative_content_env_max_boost();
    config_.window = env_int("SPEC_CONTENT_WINDOW", 8, 4, SPEC_CONTENT_WINDOW_MAX);
    config_.hysteresis = env_int("SPEC_CONTENT_HYST", 2, 1, 4);
    base_nmax_ = std::max(0, base_nmax);

    if (common_speculative_content_env_enabled()) {
        warn_env_adjustment("SPEC_CONTENT_MAX_BOOST", config_.max_boost, 0, 3);
        warn_env_adjustment("SPEC_CONTENT_WINDOW", config_.window, 4, SPEC_CONTENT_WINDOW_MAX);
        warn_env_adjustment("SPEC_CONTENT_HYST", config_.hysteresis, 1, 4);
    }

    states_.clear();
    request_stats_.clear();
    token_flags_.clear();
    token_inline_backtick_.clear();
    fence_prompt_size_.clear();
    fence_prompt_last_.clear();
    fence_prompt_open_.clear();
    inline_prompt_open_.clear();
    if (!config_.enabled || vocab == nullptr || base_nmax_ <= 0) {
        config_.enabled = false;
        return;
    }

    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    if (n_vocab <= 0) {
        config_.enabled = false;
        return;
    }

    token_flags_.resize((size_t) n_vocab, STF_NONE);
    token_inline_backtick_.resize((size_t) n_vocab, 0);
    for (int32_t token = 0; token < n_vocab; ++token) {
        const llama_token_attr attr = llama_vocab_get_attr(vocab, token);
        if (!common_speculative_content_token_attr_eligible(attr)) {
            continue;
        }
        const char * text = llama_vocab_get_text(vocab, token);
        token_flags_[(size_t) token] = content_piece_flags(text);
        token_inline_backtick_[(size_t) token] = content_inline_backtick_parity(text);
    }
    states_.resize(n_seq);
    request_stats_.resize(n_seq);
    fence_prompt_size_.resize(n_seq, 0);
    fence_prompt_last_.resize(n_seq, LLAMA_TOKEN_NULL);
    fence_prompt_open_.resize(n_seq, 0);
    inline_prompt_open_.resize(n_seq, 0);

    LOG_INF("spec content stacked verification enabled: K4V max=neural+%d window=%d hysteresis=%d provisional=%d adaptive=%d\n",
            config_.max_boost, config_.window, config_.hysteresis,
            config_.provisional ? 1 : 0, config_.adaptive ? 1 : 0);
    if (config_.provisional) {
        LOG_WRN("spec content provisional scoring is telemetry-only for batched sidecar providers\n");
    }
    if (config_.trace) {
        LOG_WRN("spec content per-cycle trace is diagnostic and must be disabled for performance measurements\n");
    }
    if (config_.adaptive) {
        LOG_WRN("spec content adaptive weighting is reserved; fixed weights remain active\n");
    }
}

common_speculative_content_state * common_speculative_content::state(uint32_t seq_id) {
    return seq_id < states_.size() ? &states_[seq_id] : nullptr;
}

const common_speculative_content_state * common_speculative_content::state(uint32_t seq_id) const {
    return seq_id < states_.size() ? &states_[seq_id] : nullptr;
}

void common_speculative_content::begin(
        uint32_t seq_id, const llama_token * tokens, size_t n_tokens) {
    auto * current = state(seq_id);
    if (current == nullptr) {
        return;
    }
    *current = common_speculative_content_state{};
    if (seq_id < request_stats_.size()) {
        request_stats_[seq_id] = {};
    }

    const size_t first = n_tokens > (size_t) config_.window
            ? n_tokens - (size_t) config_.window : 0;
    uint8_t fence_open = 0;
    uint8_t inline_open = 0;
    uint8_t fence_at_window_start = 0;
    uint8_t inline_at_window_start = 0;
    if (tokens != nullptr) {
        for (size_t i = 0; i < n_tokens; ++i) {
            if (i == first) {
                fence_at_window_start = fence_open;
                inline_at_window_start = inline_open;
            }
            advance_modes(fence_open, inline_open, tokens[i]);
        }
    }
    if (seq_id < fence_prompt_open_.size()) {
        fence_prompt_open_[seq_id] = fence_open;
        inline_prompt_open_[seq_id] = inline_open;
        fence_prompt_size_[seq_id] = tokens != nullptr ? n_tokens : 0;
        fence_prompt_last_[seq_id] = tokens != nullptr && n_tokens > 0
                ? tokens[n_tokens - 1] : LLAMA_TOKEN_NULL;
    }
    if (tokens == nullptr) {
        return;
    }

    current->fence_open = fence_at_window_start;
    current->inline_code_open = inline_at_window_start;
    for (size_t i = first; i < n_tokens; ++i) {
        push(*current, tokens[i]);
    }
}

uint8_t common_speculative_content::prepare(
        uint32_t seq_id, const llama_token * tokens, size_t n_tokens,
        llama_token id_last) {
    auto * current = state(seq_id);
    if (current == nullptr) {
        return SPEC_BOOST_0;
    }

    bool fence_before_last = false;
    bool inline_before_last = false;
    if (seq_id < fence_prompt_open_.size()) {
        size_t & tracked = fence_prompt_size_[seq_id];
        llama_token & tracked_last = fence_prompt_last_[seq_id];
        uint8_t & fence_open = fence_prompt_open_[seq_id];
        uint8_t & inline_open = inline_prompt_open_[seq_id];
        const bool prefix_changed = tracked > n_tokens ||
                (tokens != nullptr && tracked > 0 && tokens[tracked - 1] != tracked_last) ||
                (tokens == nullptr && n_tokens > 0);
        size_t first_new = tracked;
        if (prefix_changed) {
            fence_open = 0;
            inline_open = 0;
            first_new = 0;
        }
        if (tokens != nullptr) {
            for (size_t i = first_new; i < n_tokens; ++i) {
                advance_modes(fence_open, inline_open, tokens[i]);
            }
            tracked = n_tokens;
            tracked_last = n_tokens > 0 ? tokens[n_tokens - 1] : LLAMA_TOKEN_NULL;
        } else {
            tracked = 0;
            tracked_last = LLAMA_TOKEN_NULL;
        }
        fence_before_last = fence_open != 0;
        inline_before_last = inline_open != 0;
    }
    const bool id_last_is_fence = (token_flags(id_last) & STF_FENCE) != 0;
    uint8_t fence_after_last = fence_before_last ? 1 : 0;
    uint8_t inline_after_last = inline_before_last ? 1 : 0;
    if (id_last >= 0) {
        advance_modes(fence_after_last, inline_after_last, id_last);
    }

    // dp.prompt excludes id_last. Reserve one position for it so the
    // effective evidence window is exactly SPEC_CONTENT_WINDOW, not window+1.
    const size_t n_tail = id_last >= 0 ? 1 : 0;
    const size_t prompt_keep = (size_t) config_.window > n_tail
            ? (size_t) config_.window - n_tail : 0;
    const size_t prompt_first = n_tokens > prompt_keep
            ? n_tokens - prompt_keep : 0;
    uint8_t fence_at_window_start = fence_before_last ? 1 : 0;
    uint8_t inline_at_window_start = inline_before_last ? 1 : 0;
    if (tokens != nullptr) {
        rewind_modes(fence_at_window_start, inline_at_window_start,
                tokens + prompt_first, n_tokens - prompt_first);
    }

    const uint8_t previous_level = current->current_level;
    *current = common_speculative_content_state{};
    current->current_level = (spec_content_boost_level)
            std::min(previous_level, (uint8_t) config_.max_boost);
    current->candidate_level = current->current_level;
    current->fence_open = fence_at_window_start ? 1 : 0;
    current->inline_code_open = inline_at_window_start ? 1 : 0;

    if (tokens != nullptr && prompt_keep > 0) {
        for (size_t i = prompt_first; i < n_tokens; ++i) {
            push(*current, tokens[i]);
        }
    }
    if (id_last >= 0) {
        push(*current, id_last);
    }

    current->fence_open = fence_after_last;
    current->inline_code_open = inline_after_last;
    if (fence_after_last != 0) {
        current->structure_score = std::max<int16_t>(current->structure_score, 8);
        update_level(*current, SPEC_BOOST_2, true);
    } else if (fence_before_last && id_last_is_fence) {
        // A closing fence transitions immediately back to local evidence;
        // never treat it as a fresh opening anchor for trailing prose.
        current->structure_score = std::min<int16_t>(current->structure_score, 0);
        update_level(*current, SPEC_BOOST_0, false);
    }

    current->level_before = level(*current);
    current->level_at_base = current->level_before;
    current->level_final = current->level_before;
    current->cycle_valid = false;
    return current->level_before;
}

uint16_t common_speculative_content::token_flags(llama_token token) const {
    if (token < 0 || (size_t) token >= token_flags_.size()) {
        return STF_NONE;
    }
    return token_flags_[(size_t) token];
}

bool common_speculative_content::token_inline_backtick(llama_token token) const {
    return token >= 0 && (size_t) token < token_inline_backtick_.size() &&
            token_inline_backtick_[(size_t) token] != 0;
}

void common_speculative_content::advance_modes(
        uint8_t & fence_open, uint8_t & inline_open, llama_token token) const {
    if ((token_flags(token) & STF_FENCE) != 0) {
        fence_open ^= 1;
        inline_open = 0;
    } else if (fence_open == 0 && token_inline_backtick(token)) {
        inline_open ^= 1;
    }
}

void common_speculative_content::rewind_modes(
        uint8_t & fence_open, uint8_t & inline_open,
        const llama_token * tokens, size_t n_tokens) const {
    if (tokens == nullptr) return;
    for (size_t i = n_tokens; i > 0; --i) {
        const llama_token token = tokens[i - 1];
        if ((token_flags(token) & STF_FENCE) != 0) {
            fence_open ^= 1;
            // A fence transition intentionally resets inline mode. Its earlier
            // value is not invertible, so choose the conservative closed state;
            // inline evidence before the delimiter cannot leak past it.
            inline_open = 0;
        } else if (fence_open == 0 && token_inline_backtick(token)) {
            inline_open ^= 1;
        }
    }
}

void common_speculative_content::push(
        common_speculative_content_state & current, llama_token token) const {
    const uint16_t flags = token_flags(token);
    const bool inline_tick = token_inline_backtick(token);
    const bool inline_was_open = current.inline_code_open != 0;
    const bool fence_marker = (flags & STF_FENCE) != 0;
    const bool fence_was_open = current.fence_open != 0;
    const bool inline_marker = inline_tick && !fence_marker && !fence_was_open;
    const bool closing_fence = fence_marker && fence_was_open;
    const bool closing_inline = inline_marker && inline_was_open;
    const spec_content_roll_entry * prev0 = history_back(
            current, 0, (uint8_t) config_.window);

    const uint16_t structural = STF_BRACE | STF_BRACKET | STF_PAREN |
            STF_QUOTE | STF_COLON | STF_COMMA | STF_OPERATOR |
            STF_CODE_HINT | STF_MATH_HINT | STF_FENCE | STF_COMMENT_HINT;
    const bool syntax_cluster = prev0 != nullptr &&
            ((prev0->flags & (STF_CODE_HINT | STF_OPERATOR | STF_BRACE | STF_BRACKET)) != 0) &&
            ((flags & (STF_CODE_HINT | STF_OPERATOR | STF_BRACE | STF_BRACKET)) != 0);
    bool local_code_context = fence_was_open;
    for (uint8_t i = 0; !local_code_context && i < current.history_size; ++i) {
        const auto * entry = history_back(current, i, (uint8_t) config_.window);
        if (entry != nullptr &&
                (entry->flags & (STF_CODE_HINT | STF_COMMENT_HINT | STF_BRACE |
                                 STF_BRACKET | STF_FENCE)) != 0 &&
                !(entry->token >= 0 &&
                  (size_t) entry->token < token_inline_backtick_.size() &&
                  token_inline_backtick_[entry->token])) {
            local_code_context = true;
        }
    }

    int delta = 0;
    if ((flags & STF_BRACE) != 0) delta += 3;
    if ((flags & STF_BRACKET) != 0) delta += 3;
    if ((flags & STF_PAREN) != 0 && local_code_context) delta += 2;
    if ((flags & STF_CODE_HINT) != 0 && !inline_marker) delta += 3;
    if ((flags & STF_OPERATOR) != 0) delta += local_code_context ? 2 : 1;
    if ((flags & STF_NEWLINE) != 0) delta += 1;
    if ((flags & STF_COMMENT_HINT) != 0) delta += 3;
    if ((flags & STF_STRONG_ANCHOR) != 0 && !(fence_marker && fence_was_open)) delta += 3;
    // Short inline identifiers are common in prose reasoning. They are
    // delimiters and closing barriers, but not positive width evidence by
    // themselves; sustained syntax inside them can still accumulate normally.
    if (inline_marker && inline_was_open) delta -= 4;
    if (syntax_cluster) delta += 2;
    if ((flags & STF_DIGIT) != 0 && prev0 != nullptr &&
            (prev0->flags & STF_DIGIT) != 0) delta += 1;

    if ((flags & STF_ALPHA) != 0 && (flags & structural) == 0) {
        delta -= 1;
    }
    if ((flags & STF_WHITESPACE) != 0 && (flags & STF_NEWLINE) == 0 &&
            (flags & structural) == 0 && (flags & STF_ALPHA) == 0) {
        delta -= 1;
    }

    current.structure_score = (int16_t) clamp_score(current.structure_score + delta);

    const bool strong_anchor =
            (flags & STF_STRONG_ANCHOR) != 0 && !closing_fence;
    if (closing_fence || closing_inline) {
        // Closing delimiters are evidence barriers. Preceding code must not
        // re-boost explanatory prose merely because it remains in the window.
        current.structure_score = 0;
        update_level(current, SPEC_BOOST_0, false);
    } else {
        update_level(current, score_level(current.structure_score), strong_anchor);
    }
    advance_modes(current.fence_open, current.inline_code_open, token);

    const spec_content_roll_entry entry = { token, flags };
    const uint8_t capacity = (uint8_t) config_.window;
    if (current.history_size < capacity) {
        current.history[current.history_size++] = entry;
        current.history_pos = current.history_size % capacity;
    } else {
        current.history[current.history_pos] = entry;
        current.history_pos = (current.history_pos + 1) % capacity;
    }
}

void common_speculative_content::extend_window(
        common_speculative_content_state & current,
        const llama_token * tokens, size_t n_tokens) const {
    if (tokens == nullptr || n_tokens == 0) {
        return;
    }

    llama_token merged[SPEC_CONTENT_WINDOW_MAX] = {};
    const size_t capacity = (size_t) config_.window;
    const size_t n_append = std::min(n_tokens, capacity);
    const size_t n_keep = std::min<size_t>(
            current.history_size, capacity - n_append);

    // Copy the retained committed suffix oldest-to-newest, then the newest
    // extension suffix. Rebuilding keeps nonlinear pair/repetition evidence
    // exact when the fixed-size rolling window evicts old tokens.
    for (size_t i = 0; i < n_keep; ++i) {
        const auto * entry = history_back(
                current, n_keep - i - 1, (uint8_t) capacity);
        merged[i] = entry != nullptr ? entry->token : LLAMA_TOKEN_NULL;
    }
    const size_t first_append = n_tokens - n_append;
    for (size_t i = 0; i < n_append; ++i) {
        merged[n_keep + i] = tokens[first_append + i];
    }

    uint8_t fence_at_window_start = current.fence_open;
    uint8_t inline_at_window_start = current.inline_code_open;
    rewind_modes(fence_at_window_start, inline_at_window_start,
            merged, n_keep);
    for (size_t i = 0; i < first_append; ++i) {
        advance_modes(fence_at_window_start, inline_at_window_start, tokens[i]);
    }

    const uint8_t previous_level = current.current_level;
    current = common_speculative_content_state{};
    current.current_level = (spec_content_boost_level)
            std::min(previous_level, (uint8_t) config_.max_boost);
    current.candidate_level = current.current_level;
    current.fence_open = fence_at_window_start ? 1 : 0;
    current.inline_code_open = inline_at_window_start ? 1 : 0;
    for (size_t i = 0; i < n_keep + n_append; ++i) {
        push(current, merged[i]);
    }
}

uint8_t common_speculative_content::score_level(int score) const {
    if (score >= SPEC_CONTENT_SCORE_LEVEL_3) return SPEC_BOOST_3;
    if (score >= SPEC_CONTENT_SCORE_LEVEL_2) return SPEC_BOOST_2;
    if (score >= SPEC_CONTENT_SCORE_LEVEL_1) return SPEC_BOOST_1;
    return SPEC_BOOST_0;
}

void common_speculative_content::update_level(
        common_speculative_content_state & current, uint8_t desired,
        bool strong_anchor) const {
    desired = std::min<uint8_t>(desired, (uint8_t) config_.max_boost);
    const uint8_t old = current.current_level;

    if (desired == old) {
        current.candidate_level = (spec_content_boost_level) desired;
        current.candidate_age = 0;
        return;
    }

    if (desired < old) {
        current.current_level = (spec_content_boost_level) desired;
        current.candidate_level = (spec_content_boost_level) desired;
        current.candidate_age = 0;
        return;
    }

    if (strong_anchor) {
        current.current_level = (spec_content_boost_level) std::min<uint8_t>(desired, old + 1);
        current.candidate_level = current.current_level;
        current.candidate_age = 0;
        return;
    }

    if (current.candidate_level == desired) {
        current.candidate_age = std::min<uint8_t>(255, current.candidate_age + 1);
    } else {
        current.candidate_level = (spec_content_boost_level) desired;
        current.candidate_age = 1;
    }
    if (current.candidate_age >= (uint8_t) config_.hysteresis) {
        current.current_level = (spec_content_boost_level) std::min<uint8_t>(desired, old + 1);
        current.candidate_age = 0;
    }
}

uint8_t common_speculative_content::level(
        const common_speculative_content_state & current) const {
    return std::min<uint8_t>(current.current_level, (uint8_t) config_.max_boost);
}

int common_speculative_content::selected_limit(
        int base_nmax, int context_limit, bool user_override,
        uint8_t level_before) const {
    if (base_nmax <= 0) {
        return 0;
    }
    const int context = context_limit > 0 ? context_limit : base_nmax;
    if (user_override) {
        return std::max(0, context_limit);
    }
    if (!config_.enabled) {
        return std::min(base_nmax, context);
    }
    const int64_t selected = (int64_t) base_nmax +
            std::min<int>(level_before, config_.max_boost);
    return (int) std::min<int64_t>(selected, context);
}

void common_speculative_content::observe_draft(
        uint32_t seq_id, const llama_token * tokens, size_t n_tokens,
        int base_nmax, int selected_nmax, bool used_k4v) {
    auto * current = state(seq_id);
    if (current == nullptr) {
        return;
    }

    current->level_at_base = current->level_before;
    current->level_final = current->level_before;
    if (config_.provisional || config_.trace) {
        common_speculative_content_state provisional = *current;
        const size_t limit = std::min<size_t>(n_tokens, SPEC_CONTENT_WINDOW_MAX + 3);
        const size_t base_count = std::min<size_t>(limit, std::max(0, base_nmax));
        extend_window(provisional, tokens, base_count);
        current->level_at_base = level(provisional);
        extend_window(provisional,
                tokens != nullptr ? tokens + base_count : nullptr,
                limit - base_count);
        current->level_final = level(provisional);
    }
    current->base_nmax = base_nmax;
    current->selected_nmax = selected_nmax;
    current->used_k4v = used_k4v;

    // Selection raises only K4V's ceiling. A neural fallback remains a base
    // cycle, and a short K4V result pays no wider target-verification pass.
    if (selected_nmax < base_nmax) {
        current->final_nmax = selected_nmax;
    } else if (used_k4v) {
        const int32_t drafted_width = (int32_t) std::min<size_t>(
                n_tokens, (size_t) INT32_MAX);
        current->final_nmax = std::min(
                selected_nmax, std::max(base_nmax, drafted_width));
    } else {
        current->final_nmax = base_nmax;
    }
    current->boost_used = (uint8_t) std::clamp(current->final_nmax - base_nmax, 0, 3);
    current->drafted = (uint32_t) n_tokens;
    current->cycle_valid = true;
    current->stats_valid = selected_nmax >= base_nmax;
}

void common_speculative_content::accept(
        uint32_t seq_id, const llama_token * tokens, size_t n_tokens,
        uint16_t n_committed) {
    accept(seq_id, tokens, n_tokens, n_committed, n_committed);
}

void common_speculative_content::accept(
        uint32_t seq_id, const llama_token * tokens, size_t n_tokens,
        uint16_t n_committed, uint16_t n_accepted_draft) {
    auto * current = state(seq_id);
    if (current == nullptr || !current->cycle_valid) {
        return;
    }

    const common_speculative_content_state cycle = *current;
    common_speculative_content_state after = *current;
    const size_t committed = std::min<size_t>(n_committed, n_tokens);
    const size_t accepted_draft = std::min<size_t>(
            n_accepted_draft, std::min<size_t>(committed, cycle.drafted));
    extend_window(after, tokens, committed);
    const uint8_t level_after = level(after);

    const uint8_t bucket = std::min<uint8_t>(cycle.boost_used, 3);
    auto record = [accepted_draft, &cycle](common_speculative_content_stats & stats) {
        ++stats.cycles;
        stats.drafted += cycle.drafted;
        stats.accepted += accepted_draft;
        const size_t n_pos = std::min<size_t>(accepted_draft, stats.accepted_per_pos.size());
        for (size_t i = 0; i < n_pos; ++i) ++stats.accepted_per_pos[i];
    };
    if (cycle.stats_valid) {
        record(stats_[bucket]);
        if (seq_id < request_stats_.size()) {
            record(request_stats_[seq_id][bucket]);
        }
    }

    if (config_.trace) {
        LOG_INF("spec content stacked cycle seq=%u provider=%s level_before=%u level_base=%u level_provisional=%u level_after=%u neural_nmax=%d selected_k4v_nmax=%d verification_nmax=%d boost=%u drafted=%u accepted_draft=%zu committed=%zu stats_valid=%d\n",
                seq_id, cycle.used_k4v ? "k4v" : "neural",
                cycle.level_before, cycle.level_at_base,
                cycle.level_final, level_after, cycle.base_nmax,
                cycle.selected_nmax, cycle.final_nmax, cycle.boost_used,
                cycle.drafted, accepted_draft, committed,
                cycle.stats_valid ? 1 : 0);
    }

    // Commit every verified context token (including a replayed target
    // replacement), but count only actual accepted draft tokens in telemetry.
    // Rejected/provisional tokens existed only in the local copy above.
    *current = after;
    current->cycle_valid = false;
}

void common_speculative_content::reset(uint32_t seq_id) {
    if (auto * current = state(seq_id)) {
        *current = common_speculative_content_state{};
    }
    if (seq_id < fence_prompt_open_.size()) {
        fence_prompt_open_[seq_id] = 0;
        inline_prompt_open_[seq_id] = 0;
        fence_prompt_size_[seq_id] = 0;
        fence_prompt_last_[seq_id] = LLAMA_TOKEN_NULL;
    }
}

void common_speculative_content::print_stats(int32_t seq_id) const {
    if (!config_.enabled) {
        return;
    }
    const bool request_scope = seq_id >= 0 && (size_t) seq_id < request_stats_.size();
    for (int i = 0; i < 4; ++i) {
        const auto & stats = request_scope ? request_stats_[(size_t) seq_id][i] : stats_[i];
        if (stats.cycles == 0) {
            continue;
        }
        const double acceptance = stats.drafted > 0
                ? (double) stats.accepted / (double) stats.drafted : 0.0;
        const double mean_span = 1.0 + (double) stats.accepted / (double) stats.cycles;
        const int width = base_nmax_ + i;
        const int last_pos = std::clamp(width - 1, 0, (int) stats.accepted_per_pos.size() - 1);
        const double last_pos_acceptance =
                (double) stats.accepted_per_pos[(size_t) last_pos] / (double) stats.cycles;
        LOG_INF("spec content %sverification_width=%d (+%d over neural): cycles=%llu drafted=%llu accepted=%llu acceptance=%.5f mean_span=%.2f last_pos_acceptance=%.5f\n",
                request_scope ? "request " : "lifetime ", width, i,
                (unsigned long long) stats.cycles,
                (unsigned long long) stats.drafted,
                (unsigned long long) stats.accepted,
                acceptance, mean_span, last_pos_acceptance);
    }
}
