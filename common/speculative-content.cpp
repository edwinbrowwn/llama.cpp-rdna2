#include "speculative-content.h"

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
    return std::clamp((int) parsed, low, high);
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

static uint16_t classify_piece(const char * text) {
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
        if (std::strchr("=+-*/^<>!&|%~?;", c) != nullptr) flags |= STF_OPERATOR;
    }

    static const char * const code_words[] = {
        "if", "else", "for", "while", "return", "def", "class", "function",
        "const", "let", "var", "import", "struct", "enum", "switch", "case",
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
        flags |= STF_CODE_HINT;
    }

    if (contains_text(text, "\\frac") || contains_text(text, "\\sum") ||
            contains_text(text, "\\sqrt") || contains_text(text, "\\int")) {
        flags |= STF_MATH_HINT;
    }

    if (contains_text(text, "```") || contains_text(text, "~~~")) {
        flags |= STF_FENCE;
    }
    if (contains_text(text, "//") || contains_text(text, "/*") ||
            contains_text(text, "*/") || starts_with_hash_comment(text)) {
        flags |= STF_COMMENT_HINT;
    }

    if ((flags & (STF_BRACE | STF_BRACKET | STF_FENCE | STF_COMMENT_HINT)) != 0) {
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

bool common_speculative_content_env_enabled() {
    return env_switch_enabled("SPEC_CONTENT_BOOST") &&
            common_speculative_content_env_max_boost() > 0;
}

int common_speculative_content_env_max_boost() {
    return env_int("SPEC_CONTENT_MAX_BOOST", 3, 0, 3);
}

void common_speculative_content::init(
        const llama_vocab * vocab, uint32_t n_seq, bool sidecar_only, int base_nmax) {
    config_ = {};
    config_.enabled = sidecar_only && common_speculative_content_env_enabled();
    config_.provisional = env_switch_enabled("SPEC_CONTENT_PROVISIONAL");
    config_.trace = env_switch_enabled("SPEC_CONTENT_TRACE");
    config_.adaptive = env_switch_enabled("SPEC_CONTENT_ADAPT");
    config_.max_boost = common_speculative_content_env_max_boost();
    config_.window = env_int("SPEC_CONTENT_WINDOW", 12, 4, SPEC_CONTENT_WINDOW_MAX);
    config_.hysteresis = env_int("SPEC_CONTENT_HYST", 2, 1, 4);
    base_nmax_ = std::max(0, base_nmax);

    if (env_switch_enabled("SPEC_CONTENT_BOOST")) {
        warn_env_adjustment("SPEC_CONTENT_MAX_BOOST", config_.max_boost, 0, 3);
        warn_env_adjustment("SPEC_CONTENT_WINDOW", config_.window, 4, SPEC_CONTENT_WINDOW_MAX);
        warn_env_adjustment("SPEC_CONTENT_HYST", config_.hysteresis, 1, 4);
    }

    states_.clear();
    token_flags_.clear();
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
    for (int32_t token = 0; token < n_vocab; ++token) {
        token_flags_[(size_t) token] = classify_piece(llama_vocab_get_text(vocab, token));
    }
    states_.resize(n_seq);

    LOG_INF("spec content boost POC enabled for sidecar: max=+%d window=%d hysteresis=%d provisional=%d adaptive=%d\n",
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
    *current = {};
    if (tokens == nullptr) {
        return;
    }

    const size_t first = n_tokens > (size_t) config_.window
            ? n_tokens - (size_t) config_.window : 0;
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

    const uint8_t previous_level = current->current_level;
    *current = {};
    current->current_level = (spec_content_boost_level)
            std::min(previous_level, (uint8_t) config_.max_boost);
    current->candidate_level = current->current_level;

    // dp.prompt excludes id_last. Reserve one position for it so the
    // effective evidence window is exactly SPEC_CONTENT_WINDOW, not window+1.
    const size_t n_tail = id_last >= 0 ? 1 : 0;
    const size_t prompt_keep = (size_t) config_.window > n_tail
            ? (size_t) config_.window - n_tail : 0;
    if (tokens != nullptr && prompt_keep > 0) {
        const size_t first = n_tokens > prompt_keep
                ? n_tokens - prompt_keep : 0;
        for (size_t i = first; i < n_tokens; ++i) {
            push(*current, tokens[i]);
        }
    }
    if (id_last >= 0) {
        push(*current, id_last);
    }

    current->level_before = level(*current);
    current->level_at_base = current->level_before;
    current->level_final = current->level_before;
    current->level_after = current->level_before;
    current->cycle_valid = false;
    return current->level_before;
}

uint16_t common_speculative_content::token_flags(llama_token token) const {
    if (token < 0 || (size_t) token >= token_flags_.size()) {
        return STF_NONE;
    }
    return token_flags_[(size_t) token];
}

void common_speculative_content::push(
        common_speculative_content_state & current, llama_token token) const {
    const uint16_t flags = token_flags(token);
    const spec_content_roll_entry * prev0 = history_back(current, 0, (uint8_t) config_.window);
    const spec_content_roll_entry * prev1 = history_back(current, 1, (uint8_t) config_.window);
    const spec_content_roll_entry * prev2 = history_back(current, 2, (uint8_t) config_.window);
    const spec_content_roll_entry * prev3 = history_back(current, 3, (uint8_t) config_.window);
    const spec_content_roll_entry * prev4 = history_back(current, 4, (uint8_t) config_.window);

    const bool quote_colon =
            ((flags & STF_COLON) != 0 && current.quote_recent != 0) ||
            ((flags & STF_QUOTE) != 0 && current.colon_recent != 0);
    const bool repeat2 = prev0 != nullptr && prev1 != nullptr && prev2 != nullptr &&
            prev0->token == prev2->token && token == prev1->token;
    const bool repeat3 = prev0 != nullptr && prev1 != nullptr && prev2 != nullptr &&
            prev3 != nullptr && prev4 != nullptr &&
            prev1->token == prev4->token && prev0->token == prev3->token &&
            token == prev2->token;

    const uint16_t structural = STF_BRACE | STF_BRACKET | STF_PAREN |
            STF_QUOTE | STF_COLON | STF_COMMA | STF_OPERATOR |
            STF_CODE_HINT | STF_MATH_HINT | STF_FENCE | STF_COMMENT_HINT;
    const bool syntax_cluster = prev0 != nullptr &&
            ((prev0->flags & (STF_CODE_HINT | STF_OPERATOR | STF_BRACE | STF_BRACKET)) != 0) &&
            ((flags & (STF_CODE_HINT | STF_OPERATOR | STF_BRACE | STF_BRACKET)) != 0);

    int delta = 0;
    if ((flags & STF_BRACE) != 0) delta += 3;
    if ((flags & STF_BRACKET) != 0) delta += 3;
    if ((flags & STF_PAREN) != 0) delta += 2;
    if ((flags & STF_CODE_HINT) != 0) delta += 3;
    if ((flags & STF_MATH_HINT) != 0) delta += 2;
    if ((flags & STF_OPERATOR) != 0) delta += 2;
    if ((flags & STF_NEWLINE) != 0) delta += 1;
    if ((flags & STF_STRONG_ANCHOR) != 0) delta += 8;
    if (quote_colon) delta += 4;
    if (syntax_cluster) delta += 2;
    if (repeat2 || repeat3) delta += 4;
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
    if (repeat3) {
        current.repetition_score = std::min<uint8_t>(3, current.repetition_score + 2);
    } else if (repeat2) {
        current.repetition_score = std::min<uint8_t>(3, current.repetition_score + 1);
    } else if (current.repetition_score > 0) {
        --current.repetition_score;
    }

    const bool brace = (flags & STF_BRACE) != 0;
    const bool bracket = (flags & STF_BRACKET) != 0;
    const bool paren = (flags & STF_PAREN) != 0;
    current.brace_depth = brace ? std::min<uint8_t>(8, current.brace_depth + 1)
                                : (current.brace_depth > 0 ? current.brace_depth - 1 : 0);
    current.bracket_depth = bracket ? std::min<uint8_t>(8, current.bracket_depth + 1)
                                    : (current.bracket_depth > 0 ? current.bracket_depth - 1 : 0);
    current.paren_depth = paren ? std::min<uint8_t>(8, current.paren_depth + 1)
                                : (current.paren_depth > 0 ? current.paren_depth - 1 : 0);

    current.quote_recent = (flags & STF_QUOTE) != 0 ? 2 : (current.quote_recent > 0 ? current.quote_recent - 1 : 0);
    current.colon_recent = (flags & STF_COLON) != 0 ? 2 : (current.colon_recent > 0 ? current.colon_recent - 1 : 0);
    current.newline_recent = (flags & STF_NEWLINE) != 0 ? 2 : (current.newline_recent > 0 ? current.newline_recent - 1 : 0);
    current.operator_recent = (flags & STF_OPERATOR) != 0 ? 2 : (current.operator_recent > 0 ? current.operator_recent - 1 : 0);

    const bool strong_anchor = (flags & STF_STRONG_ANCHOR) != 0;
    update_level(current, score_level(current.structure_score), strong_anchor,
            repeat2 || repeat3);

    const spec_content_roll_entry entry = { token, flags };
    const uint8_t capacity = (uint8_t) config_.window;
    if (current.history_size < capacity) {
        current.history[current.history_size++] = entry;
        current.history_pos = current.history_size % capacity;
    } else {
        current.history[current.history_pos] = entry;
        current.history_pos = (current.history_pos + 1) % capacity;
    }
    current.token_count = current.token_count == UINT32_MAX
            ? UINT32_MAX : current.token_count + 1;
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

    const uint8_t previous_level = current.current_level;
    current = {};
    current.current_level = (spec_content_boost_level)
            std::min(previous_level, (uint8_t) config_.max_boost);
    current.candidate_level = current.current_level;
    for (size_t i = 0; i < n_keep + n_append; ++i) {
        push(current, merged[i]);
    }
}

uint8_t common_speculative_content::score_level(int score) const {
    if (score >= 12) return SPEC_BOOST_3;
    if (score >= 8)  return SPEC_BOOST_2;
    if (score >= 4)  return SPEC_BOOST_1;
    return SPEC_BOOST_0;
}

void common_speculative_content::update_level(
        common_speculative_content_state & current, uint8_t desired,
        bool strong_anchor, bool repeated) const {
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
        current.transition_cooldown = 0;
        return;
    }

    if (strong_anchor || repeated) {
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

int common_speculative_content::max_limit(
        int base_nmax, int context_limit, bool user_override) const {
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
    return std::min(base_nmax + config_.max_boost, context);
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
    return std::min(base_nmax + std::min<int>(level_before, config_.max_boost), context);
}

void common_speculative_content::observe_draft(
        uint32_t seq_id, const llama_token * tokens, size_t n_tokens,
        int base_nmax, int selected_nmax, int hard_nmax) {
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
    current->final_nmax = std::min(selected_nmax, hard_nmax);
    current->boost_used = (uint8_t) std::clamp(current->final_nmax - base_nmax, 0, 3);
    current->drafted = (uint32_t) n_tokens;
    current->cycle_valid = true;
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
    ++stats_[bucket].cycles;
    stats_[bucket].drafted += cycle.drafted;
    stats_[bucket].accepted += accepted_draft;

    if (config_.trace) {
        LOG_INF("spec content cycle seq=%u level_before=%u level_provisional=%u level_after=%u base_nmax=%d selected_nmax=%d boost=%u drafted=%u accepted_draft=%zu committed=%zu\n",
                seq_id, cycle.level_before, cycle.level_final,
                level_after, cycle.base_nmax, cycle.final_nmax,
                cycle.boost_used, cycle.drafted, accepted_draft, committed);
    }

    // Commit every verified context token (including a replayed target
    // replacement), but count only actual accepted draft tokens in telemetry.
    // Rejected/provisional tokens existed only in the local copy above.
    *current = after;
    current->level_after = level_after;
    current->accepted = (uint32_t) accepted_draft;
    current->cycle_valid = false;
}

void common_speculative_content::reset(uint32_t seq_id) {
    if (auto * current = state(seq_id)) {
        *current = {};
    }
}

void common_speculative_content::print_stats() const {
    if (!config_.enabled) {
        return;
    }
    for (int i = 0; i < 4; ++i) {
        const auto & stats = stats_[i];
        if (stats.cycles == 0) {
            continue;
        }
        LOG_INF("spec content +%d: cycles=%llu drafted=%llu accepted=%llu\n",
                i, (unsigned long long) stats.cycles,
                (unsigned long long) stats.drafted,
                (unsigned long long) stats.accepted);
    }
}
