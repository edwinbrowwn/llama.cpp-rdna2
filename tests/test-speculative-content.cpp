#include "common.h"
#include "speculative.h"
#include "speculative-content.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "content policy test failure: %s\n", message);
        std::abort();
    }
}

static void set_test_env(const char * name, const char * value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

static void unset_test_env(const char * name) {
#if defined(_WIN32)
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

struct common_speculative_content_test_access {
    static void configure(common_speculative_content & content, int base_nmax,
            int max_boost, int window, bool provisional = false, bool trace = false,
            uint32_t n_seq = 1) {
        content.config_ = {};
        content.config_.enabled = max_boost > 0;
        content.config_.max_boost = max_boost;
        content.config_.window = window;
        content.config_.hysteresis = 2;
        content.config_.provisional = provisional;
        content.config_.trace = trace;
        content.base_nmax_ = base_nmax;
        content.token_flags_.assign(32, STF_ALPHA);
        content.token_inline_backtick_.assign(32, 0);
        content.token_flags_[5] = STF_BRACE | STF_COMMENT_HINT | STF_STRONG_ANCHOR;
        content.token_flags_[6] = STF_FENCE | STF_STRONG_ANCHOR;
        content.token_flags_[7] = STF_QUOTE | STF_CODE_HINT;
        content.token_inline_backtick_[7] = 1;
        content.token_flags_[9]  = STF_OPERATOR;
        content.token_flags_[10] = STF_PAREN;
        content.token_flags_[11] = STF_DIGIT;
        content.token_flags_[12] = STF_MATH_HINT;
        content.token_flags_[13] = STF_ALPHA | STF_CODE_HINT;
        content.token_flags_[14] = STF_NEWLINE;
        content.token_flags_[15] = STF_BRACE;
        content.states_.assign(n_seq, {});
        content.fence_prompt_size_.assign(n_seq, 0);
        content.fence_prompt_last_.assign(n_seq, LLAMA_TOKEN_NULL);
        content.fence_prompt_open_.assign(n_seq, 0);
        content.inline_prompt_open_.assign(n_seq, 0);
        content.request_stats_.assign(n_seq, {});
    }

    static const common_speculative_content_stats & stats(
            const common_speculative_content & content, int bucket) {
        return content.stats_[bucket];
    }

    static const common_speculative_content_stats & request_stats(
            const common_speculative_content & content, int bucket) {
        return content.request_stats_[0][bucket];
    }

    static std::vector<llama_token> history_tokens(
            const common_speculative_content & content, uint32_t seq_id) {
        std::vector<llama_token> result;
        const auto * state = content.state(seq_id);
        if (state == nullptr) return result;
        result.reserve(state->history_size);
        for (uint8_t i = 0; i < state->history_size; ++i) {
            const uint8_t index = state->history_size < content.config_.window
                    ? i : (uint8_t) ((state->history_pos + i) % content.config_.window);
            result.push_back(state->history[index].token);
        }
        return result;
    }
};

static void test_piece_classification_guards() {
    const uint16_t url = common_speculative_content_piece_flags("https://example.test/a?id=3");
    require((url & (STF_OPERATOR | STF_COMMENT_HINT)) == 0,
            "URL schemes cannot masquerade as operators or comments");
    const uint16_t path = common_speculative_content_piece_flags("/var");
    require((path & STF_OPERATOR) == 0,
            "filesystem path pieces cannot masquerade as operators");
    require((common_speculative_content_piece_flags("return") & STF_CODE_HINT) != 0,
            "standalone source keyword remains code evidence");
    require((common_speculative_content_piece_flags("returning") & STF_CODE_HINT) == 0,
            "source keywords require word boundaries");
    require((common_speculative_content_piece_flags("```") &
            (STF_FENCE | STF_STRONG_ANCHOR)) ==
            (STF_FENCE | STF_STRONG_ANCHOR),
            "backtick fence delimiter remains a strong anchor");
    require((common_speculative_content_piece_flags("?|%~") & STF_OPERATOR) == 0 &&
            (common_speculative_content_piece_flags("|") & STF_OPERATOR) == 0,
            "prose punctuation and single table pipes are not operators");
    require((common_speculative_content_piece_flags("||") &
            (STF_OPERATOR | STF_CODE_HINT)) ==
            (STF_OPERATOR | STF_CODE_HINT),
            "multi-character source operators retain code evidence");
    require((common_speculative_content_piece_flags("---:") &
            (STF_OPERATOR | STF_CODE_HINT)) == 0,
            "Markdown separator rules cannot trigger source-code width");
    require((common_speculative_content_piece_flags("**") & STF_OPERATOR) == 0 &&
            (common_speculative_content_piece_flags(":**") & STF_OPERATOR) == 0 &&
            (common_speculative_content_piece_flags("-to") & STF_OPERATOR) == 0,
            "Markdown emphasis and hyphenated prose are not source operators");
    require((common_speculative_content_piece_flags("->") &
            (STF_OPERATOR | STF_CODE_HINT)) ==
            (STF_OPERATOR | STF_CODE_HINT) &&
            (common_speculative_content_piece_flags("i--") & STF_OPERATOR) != 0,
            "source arrows and decrement operators remain code evidence");
    require(common_speculative_content_inline_backtick_parity("`value") &&
            !common_speculative_content_inline_backtick_parity("```cpp"),
            "inline backticks remain separate from fenced delimiters");
    require(common_speculative_content_token_attr_eligible(LLAMA_TOKEN_ATTR_NORMAL) &&
            !common_speculative_content_token_attr_eligible(LLAMA_TOKEN_ATTR_CONTROL) &&
            !common_speculative_content_token_attr_eligible(LLAMA_TOKEN_ATTR_USER_DEFINED) &&
            !common_speculative_content_token_attr_eligible(LLAMA_TOKEN_ATTR_UNUSED) &&
            !common_speculative_content_token_attr_eligible(LLAMA_TOKEN_ATTR_BYTE) &&
            !common_speculative_content_token_attr_eligible(LLAMA_TOKEN_ATTR_UNDEFINED),
            "only normal content tokens can drive the K4V verification signal");
}

static void test_math_evidence_requires_code_context() {
    common_speculative_content content;
    common_speculative_content_test_access::configure(content, 4, 1, 8);

    const llama_token formula[] = {
        0, 9, 10, 11, 9, 11, 10, 9, 11, 12, 0,
    };
    for (size_t i = 0; i < sizeof(formula) / sizeof(formula[0]); ++i) {
        require(content.prepare(0, formula, i, formula[i]) == SPEC_BOOST_0,
                "dense inline formulas remain at the prose width");
    }

    content.reset(0);
    const llama_token source[] = {
        13, 0, 9, 0, 10, 9, 14, 15, 14, 13, 0, 9,
    };
    bool widened = false;
    for (size_t i = 0; i < sizeof(source) / sizeof(source[0]); ++i) {
        widened = widened ||
                content.prepare(0, source, i, source[i]) == SPEC_BOOST_1;
    }
    require(widened, "sustained unfenced source still selects the code width");
}

static void test_candidate_defaults() {
    unset_test_env("SPEC_SIDECAR");
    unset_test_env("SPEC_CONTENT_BOOST");
    unset_test_env("SPEC_CONTENT_MAX_BOOST");
    unset_test_env("SPEC_CONTENT_WINDOW");
    require(!common_speculative_content_env_enabled(),
            "content verification cannot activate without the sidecar master gate");

    set_test_env("SPEC_SIDECAR", "true");
    require(!common_speculative_content_env_enabled(),
            "sidecar master gate retains exact SPEC_SIDECAR=1 semantics");
    set_test_env("SPEC_SIDECAR", "1");
    require(common_speculative_content_env_enabled(),
            "stacked verification defaults on with SPEC_SIDECAR=1");

    for (const char * disabled : { "0", "off", "false", "invalid" }) {
        set_test_env("SPEC_CONTENT_BOOST", disabled);
        require(!common_speculative_content_env_enabled(),
                "explicit false or invalid content switch fails closed");
    }
    for (const char * enabled : { "1", "on", "true" }) {
        set_test_env("SPEC_CONTENT_BOOST", enabled);
        require(common_speculative_content_env_enabled(),
                "legacy explicit content enable remains compatible");
    }
    unset_test_env("SPEC_CONTENT_BOOST");

    common_speculative_content content;
    content.init(nullptr, 1, true, 4);
    require(common_speculative_content_env_max_boost() == 1 &&
            content.config().max_boost == 1,
            "candidate default adds at most one K4V verification token");
    require(content.config().window == 8 && content.config().hysteresis == 2,
            "candidate defaults use the selected window and hysteresis");

    set_test_env("SPEC_CONTENT_MAX_BOOST", "2");
    set_test_env("SPEC_CONTENT_WINDOW", "16");
    set_test_env("SPEC_CONTENT_HYST", "3");
    common_speculative_content configured;
    configured.init(nullptr, 1, true, 4);
    require(configured.config().max_boost == 2 &&
            configured.config().window == 16 &&
            configured.config().hysteresis == 3,
            "explicit environment values override automatic defaults");
    unset_test_env("SPEC_CONTENT_HYST");

    set_test_env("SPEC_CONTENT_MAX_BOOST", "0");
    require(!common_speculative_content_env_enabled(),
            "zero maximum boost is an independent opt-out");
    set_test_env("SPEC_CONTENT_MAX_BOOST", "999999999999999999999999");
    set_test_env("SPEC_CONTENT_WINDOW", "-999999999999999999999999");
    common_speculative_content clamped;
    clamped.init(nullptr, 1, true, 4);
    require(clamped.config().max_boost == 3 && clamped.config().window == 4,
            "extreme environment integers clamp before narrowing to int");
    set_test_env("SPEC_CONTENT_MAX_BOOST", "3");
    unset_test_env("SPEC_CONTENT_WINDOW");
}

static void test_sidecar_capacity_reservations() {
    set_test_env("SPEC_SIDECAR", "1");
    unset_test_env("SPEC_CONTENT_BOOST");
    set_test_env("SPEC_CONTENT_MAX_BOOST", "3");

    common_params params;
    params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    params.speculative.draft.n_max = 4;
    params.speculative.draft.sidecar_candidate_ready = true;
    params.speculative.draft.content_verification_eligible = true;
    require(!common_speculative_content_candidate_eligible(params.speculative) &&
            common_context_params_to_llama(params).n_rs_seq == 4,
            "a neural-only sidecar cannot acquire content generation width");

    params.speculative.types = {
        COMMON_SPECULATIVE_TYPE_DRAFT_MTP,
        COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V,
    };
    require(common_speculative_content_candidate_eligible(params.speculative),
            "qualified MTP plus K4V enables stacked verification");
    require(common_speculative_n_max(&params.speculative) == 48,
            "configured K4V width already reserves the target output envelope");
    require(common_context_params_to_llama(params).n_rs_seq == 7,
            "stacked verification reserves recurrent rollback through base plus boost");

    params.speculative.draft.sidecar_only = true;
    params.speculative.draft.sidecar_type = COMMON_SPECULATIVE_TYPE_DRAFT_MTP;
    require(common_speculative_content_runtime_eligible(params.speculative),
            "qualified and probed MTP plus K4V enables runtime selection");
    params.speculative.draft.sidecar_type = COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;
    require(!common_speculative_content_runtime_eligible(params.speculative),
            "runtime sidecar type must match the configured neural provider");

    params.speculative.synth_len = 1.0;
    require(!common_speculative_content_candidate_eligible(params.speculative) &&
            common_context_params_to_llama(params).n_rs_seq == 4,
            "synthetic mode cannot acquire stacked rollback capacity");
    params.speculative.synth_len = -1.0;

    params.speculative.types = {
        COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH,
        COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V,
    };
    params.speculative.draft.sidecar_type = COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;
    require(!common_speculative_content_candidate_eligible(params.speculative) &&
            !common_speculative_content_runtime_eligible(params.speculative) &&
            common_context_params_to_llama(params).n_rs_seq == 4,
            "DFlash plus K4V cannot acquire the negative wider target envelope");

    params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH };
    require(!common_speculative_content_candidate_eligible(params.speculative) &&
            common_context_params_to_llama(params).n_rs_seq == 4,
            "DFlash without K4V remains fixed and cannot invoke content selection");

    params.speculative.types = {
        COMMON_SPECULATIVE_TYPE_DRAFT_MTP,
        COMMON_SPECULATIVE_TYPE_DRAFT_MTP,
        COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V,
    };
    require(!common_speculative_content_candidate_eligible(params.speculative),
            "duplicate neural providers cannot acquire content eligibility");

    params.speculative.types = {
        COMMON_SPECULATIVE_TYPE_DRAFT_MTP,
        COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V,
    };
    params.speculative.draft.sidecar_type = COMMON_SPECULATIVE_TYPE_DRAFT_MTP;
    set_test_env("SPEC_CONTENT_BOOST", "0");
    require(common_context_params_to_llama(params).n_rs_seq == 4,
            "explicit content opt-out keeps stacked MTP exactly fixed-width");
    unset_test_env("SPEC_CONTENT_BOOST");
}

static void test_relative_limits_and_override() {
    common_speculative_content content;
    common_speculative_content_test_access::configure(content, 4, 3, 12);

    require(content.selected_limit(4, 128, false, SPEC_BOOST_0) == 4,
            "level zero preserves baseline");
    require(content.selected_limit(4, 128, false, SPEC_BOOST_1) == 5,
            "level one adds one");
    require(content.selected_limit(4, 128, false, SPEC_BOOST_3) == 7,
            "level three adds three");
    require(content.selected_limit(4, 5, false, SPEC_BOOST_3) == 5,
            "context budget remains a hard ceiling");
    require(content.selected_limit(4, 6, true, SPEC_BOOST_0) == 6,
            "explicit request above baseline remains exact");
    require(content.selected_limit(4, 2, true, SPEC_BOOST_3) == 2,
            "explicit request below baseline remains exact");
    require(content.selected_limit(INT32_MAX, INT32_MAX, false, SPEC_BOOST_3) == INT32_MAX,
            "relative width arithmetic cannot overflow its context ceiling");
}

static void test_exact_window_and_accepted_commit() {
    common_speculative_content content;
    common_speculative_content_test_access::configure(content, 4, 3, 4);

    const llama_token prompt[] = { 0, 1, 2, 3 };
    content.begin(0, prompt, 4);
    content.prepare(0, prompt, 4, 4);

    auto * state = content.state(0);
    require(state != nullptr, "sequence state exists");
    require(state->history_size == 4, "id_last does not create window plus one");
    require(state->history[0].token == 1 && state->history[1].token == 2 &&
            state->history[2].token == 3 && state->history[3].token == 4,
            "prepare retains the exact newest window including id_last");

    const llama_token draft[] = { 5, 6 };
    content.observe_draft(0, draft, 2, 4, 4);
    require(state->level_final == state->level_before,
            "trace-off provisional draft does not perform unused rescoring");
    content.accept(0, draft, 2, 1);

    state = content.state(0);
    require(state->history_size == 4, "committed history stays window bounded");
    require(state->history[0].token == 2 && state->history[1].token == 3 &&
            state->history[2].token == 4 && state->history[3].token == 5,
            "only target-accepted draft prefix enters committed state");
}

static void test_maximum_32_token_window() {
    common_speculative_content content;
    common_speculative_content_test_access::configure(
            content, 4, 3, SPEC_CONTENT_WINDOW_MAX);

    std::vector<llama_token> prompt(40);
    for (size_t i = 0; i < prompt.size(); ++i) {
        prompt[i] = (llama_token) i;
    }
    content.begin(0, prompt.data(), prompt.size());

    auto * state = content.state(0);
    require(state != nullptr && state->history_size == 32,
            "maximum content window retains 32 tokens");
    require(state->history[0].token == 8 && state->history[31].token == 39,
            "maximum content window retains the newest prompt suffix");

    content.prepare(0, prompt.data(), prompt.size(), 40);
    state = content.state(0);
    require(state->history_size == 32 && state->history[0].token == 9 &&
            state->history[31].token == 40,
            "32-token prepare window includes id_last without growing to 33");
}

static void test_fence_and_inline_code_modes() {
    common_speculative_content content;
    common_speculative_content_test_access::configure(content, 4, 3, 4);

    const llama_token fenced[] = { 0, 6, 1, 1, 1, 1, 1, 1, 6 };
    uint8_t level = 0;
    for (size_t i = 0; i < 7; ++i) {
        level = content.prepare(0, fenced, i, fenced[i]);
    }
    const auto * state = content.state(0);
    require(level >= SPEC_BOOST_1 && state != nullptr && state->fence_open != 0 &&
            state->structure_score >= 8,
            "fenced mode persists after the opening marker leaves the local window");

    content.prepare(0, fenced, 8, fenced[8]);
    state = content.state(0);
    require(state != nullptr && state->fence_open == 0 &&
            state->level_before == SPEC_BOOST_0 && state->structure_score <= 0,
            "closing fence immediately returns selection to local prose evidence");
    const llama_token after_fence[] = { 0, 6, 1, 1, 1, 1, 1, 1, 6, 1, 1 };
    content.prepare(0, after_fence, 9, after_fence[9]);
    require(content.state(0)->level_before == SPEC_BOOST_0,
            "pre-close code in the local window cannot re-boost trailing prose");

    content.reset(0);
    const llama_token inline_code[] = { 0, 7, 8, 8, 7, 1 };
    require(content.prepare(0, inline_code, 1, inline_code[1]) == SPEC_BOOST_0,
            "an opening inline backtick alone does not widen prose reasoning");
    require(content.prepare(0, inline_code, 2, inline_code[2]) == SPEC_BOOST_0,
            "a short inline identifier remains at the fixed prose width");
    content.prepare(0, inline_code, 4, inline_code[4]);
    state = content.state(0);
    require(state != nullptr && state->inline_code_open == 0 &&
            state->level_before == SPEC_BOOST_0,
            "closing inline backtick ends the localized boost");
    content.prepare(0, inline_code, 5, inline_code[5]);
    state = content.state(0);
    require(state != nullptr && state->inline_code_open == 0 &&
            state->level_before == SPEC_BOOST_0 && state->structure_score <= 0,
            "a retained closing backtick is a barrier for trailing prose");
}

static void test_fence_tracking_uses_committed_prompt_only() {
    common_speculative_content content;
    common_speculative_content_test_access::configure(content, 4, 3, 4);

    const llama_token long_open_prompt[] = { 6, 1, 1, 1, 1, 1, 1, 1 };
    content.begin(0, long_open_prompt, 8);
    content.prepare(0, long_open_prompt, 8, 1);
    auto * state = content.state(0);
    require(state != nullptr && state->fence_open != 0 &&
            state->level_before >= SPEC_BOOST_1,
            "begin preserves an opening fence older than the rolling window");

    const llama_token branched_prompt[] = { 0, 2 };
    content.prepare(0, branched_prompt, 2, 1);
    state = content.state(0);
    require(state != nullptr && state->fence_open == 0,
            "a changed or truncated committed prefix rebuilds fence parity");

    content.reset(0);
    const llama_token base_prompt[] = { 0, 1, 2 };
    content.prepare(0, base_prompt, 3, 3);
    const llama_token long_commit[] = { 6, 1, 1, 1, 1, 1 };
    content.observe_draft(0, long_commit, 6, 4, 7);
    content.accept(0, long_commit, 6, 6, 6);
    state = content.state(0);
    require(state != nullptr && state->fence_open != 0,
            "accepted extension preserves a fence opener older than the retained suffix");

    content.reset(0);
    const llama_token prose_prompt[] = { 0, 1, 2 };
    content.prepare(0, prose_prompt, 3, 3);
    const llama_token rejected_draft[] = { 6, 1 };
    content.observe_draft(0, rejected_draft, 2, 4, 4);
    content.accept(0, rejected_draft, 2, 0, 0);
    content.prepare(0, prose_prompt, 3, 3);
    state = content.state(0);
    require(state != nullptr && state->fence_open == 0,
            "a rejected draft fence never enters committed classifier mode");
}

static void test_request_replacement_and_sequence_isolation() {
    common_speculative_content content;
    common_speculative_content_test_access::configure(
            content, 4, 1, 4, false, false, 2);

    const llama_token fenced[] = { 6, 1, 1, 1 };
    content.begin(0, fenced, 4);
    content.prepare(0, fenced, 4, 1);
    require(content.state(0)->fence_open != 0,
            "first request records its committed fence mode");

    const llama_token prose_same_size[] = { 0, 2, 1, 1 };
    content.begin(0, prose_same_size, 4);
    content.prepare(0, prose_same_size, 4, 1);
    require(content.state(0)->fence_open == 0,
            "begin rebuilds a same-length replacement request from committed tokens");

    content.begin(0, fenced, 4);
    content.begin(1, prose_same_size, 4);
    content.prepare(0, fenced, 4, 1);
    content.prepare(1, prose_same_size, 4, 1);
    require(content.state(0)->fence_open != 0 &&
            content.state(1)->fence_open == 0,
            "content and fence state are isolated by sequence");
    content.reset(0);
    require(content.state(0)->fence_open == 0 &&
            content.state(1)->fence_open == 0 &&
            content.state(1)->history_size != 0,
            "reset clears only the requested sequence");
}

static void test_provisional_is_separate() {
    common_speculative_content content;
    common_speculative_content_test_access::configure(content, 4, 3, 4, true);

    const llama_token prompt[] = { 0, 1, 2 };
    content.prepare(0, prompt, 3, 3);
    const llama_token draft[] = { 5, 6 };
    content.observe_draft(0, draft, 2, 4, 4);

    auto * state = content.state(0);
    require(state->level_final > state->level_before,
            "provisional mode scores strong draft evidence");
    content.accept(0, draft, 2, 0);

    state = content.state(0);
    for (size_t i = 0; i < state->history_size; ++i) {
        require(state->history[i].token != 5 && state->history[i].token != 6,
                "rejected provisional tokens never enter committed history");
    }
}

static void test_request_stats_reset_at_begin() {
    common_speculative_content content;
    common_speculative_content_test_access::configure(content, 4, 1, 4);

    const llama_token prompt[] = { 0, 1, 2 };
    content.begin(0, prompt, 3);
    content.prepare(0, prompt, 3, 3);
    const llama_token draft[] = { 4, 5, 6, 7, 8 };
    content.observe_draft(0, draft, 5, 4, 5);
    content.accept(0, draft, 5, 3, 3);

    const auto & request = common_speculative_content_test_access::request_stats(content, 1);
    require(request.cycles == 1 && request.drafted == 5 && request.accepted == 3 &&
            request.accepted_per_pos[0] == 1 && request.accepted_per_pos[1] == 1 &&
            request.accepted_per_pos[2] == 1 && request.accepted_per_pos[3] == 0,
            "request stats identify selected width and per-position acceptance");

    content.begin(0, prompt, 3);
    const auto & reset = common_speculative_content_test_access::request_stats(content, 1);
    require(reset.cycles == 0 && reset.drafted == 0 && reset.accepted == 0,
            "request stats reset independently at generation begin");
    require(common_speculative_content_test_access::stats(content, 1).cycles == 1,
            "lifetime stats survive request reset");
}

static void test_stacked_verification_width_accounting() {
    common_speculative_content content;
    common_speculative_content_test_access::configure(content, 4, 1, 8);

    const llama_token prompt[] = { 0, 1, 2 };
    const llama_token draft5[] = { 3, 4, 5, 6, 7 };
    const llama_token draft4[] = { 3, 4, 5, 6 };

    content.prepare(0, prompt, 3, 3);
    content.observe_draft(0, draft5, 5, 4, 5, false);
    require(content.state(0)->selected_nmax == 5 &&
            content.state(0)->final_nmax == 4 &&
            content.state(0)->boost_used == 0 &&
            !content.state(0)->used_k4v,
            "neural fallback remains a width-four accounting cycle");
    content.accept(0, draft5, 5, 4, 4);

    content.prepare(0, prompt, 3, 3);
    content.observe_draft(0, draft4, 4, 4, 5, true);
    require(content.state(0)->final_nmax == 4 &&
            content.state(0)->boost_used == 0 &&
            content.state(0)->used_k4v,
            "short K4V hit pays no widened verification width");
    content.accept(0, draft4, 4, 4, 4);

    content.prepare(0, prompt, 3, 3);
    content.observe_draft(0, draft5, 5, 4, 5, true);
    require(content.state(0)->final_nmax == 5 &&
            content.state(0)->boost_used == 1,
            "five-token K4V hit owns the widened verification bucket");
    content.accept(0, draft5, 5, 5, 5);

    require(common_speculative_content_test_access::stats(content, 0).cycles == 2 &&
            common_speculative_content_test_access::stats(content, 1).cycles == 1,
            "only an actual long K4V proposal is charged to the boost bucket");
}

static void test_incremental_commit_matches_fresh_rebuild() {
    common_speculative_content content;
    common_speculative_content_test_access::configure(content, 4, 1, 8);

    std::vector<llama_token> committed = { 0, 1, 2 };
    content.begin(0, committed.data(), committed.size());
    uint32_t rng = 0x12345678U;
    for (int cycle = 0; cycle < 512; ++cycle) {
        rng = rng * 1664525U + 1013904223U;
        const llama_token id_last = (llama_token) ((rng >> 28) % 8);
        content.prepare(0, committed.data(), committed.size(), id_last);
        committed.push_back(id_last);

        rng = rng * 1664525U + 1013904223U;
        const size_t n_draft = 1 + ((rng >> 27) % 10);
        std::vector<llama_token> draft(n_draft);
        for (size_t i = 0; i < n_draft; ++i) {
            rng = rng * 1664525U + 1013904223U;
            draft[i] = (llama_token) ((rng >> 28) % 8);
        }
        rng = rng * 1664525U + 1013904223U;
        const uint16_t n_accept = (uint16_t) ((rng >> 27) % (n_draft + 1));
        content.observe_draft(0, draft.data(), draft.size(), 4, 5);
        content.accept(0, draft.data(), draft.size(), n_accept, n_accept);
        committed.insert(committed.end(), draft.begin(), draft.begin() + n_accept);

        // Compare the next-cycle reconstruction, not the raw post-accept
        // score: an open committed fence deliberately applies its score floor
        // in prepare(), after the current target token is known.
        common_speculative_content actual_probe = content;
        common_speculative_content fresh_probe;
        common_speculative_content_test_access::configure(fresh_probe, 4, 1, 8);
        const size_t prefix = committed.size() - 1;
        actual_probe.prepare(0, committed.data(), prefix, committed.back());
        fresh_probe.prepare(0, committed.data(), prefix, committed.back());
        const auto * actual = actual_probe.state(0);
        const auto * fresh = fresh_probe.state(0);
        require(actual != nullptr && fresh != nullptr,
                "incremental and reference states exist");
        require(common_speculative_content_test_access::history_tokens(actual_probe, 0) ==
                    common_speculative_content_test_access::history_tokens(fresh_probe, 0) &&
                actual->structure_score == fresh->structure_score &&
                actual->fence_open == fresh->fence_open &&
                actual->inline_code_open == fresh->inline_code_open,
                "incremental next-cycle state matches a fresh committed-context rebuild");
    }
}

static void test_context_limited_cycles_do_not_pollute_width_stats() {
    common_speculative_content content;
    common_speculative_content_test_access::configure(content, 4, 1, 4);

    const llama_token prompt[] = { 0, 1, 2 };
    content.prepare(0, prompt, 3, 3);
    const llama_token draft[] = { 4, 5 };
    content.observe_draft(0, draft, 2, 4, 2);
    content.accept(0, draft, 2, 1, 1);

    require(common_speculative_content_test_access::stats(content, 0).cycles == 0,
            "context-limited tail is excluded from configured-width telemetry");
    const auto * state = content.state(0);
    require(state != nullptr && state->history_size == 4 &&
            state->history[3].token == 4,
            "context-limited accepted tokens still commit to classifier state");
}

static void test_replay_commit_and_draft_accounting_are_separate() {
    common_speculative_content content;
    common_speculative_content_test_access::configure(content, 4, 3, 4);

    const llama_token prompt[] = { 0, 1, 2 };
    content.prepare(0, prompt, 3, 3);
    const llama_token replay[] = { 4, 5 };
    content.observe_draft(0, replay, 2, 4, 4);
    content.accept(0, replay, 2, 2, 1);

    const auto & stats = common_speculative_content_test_access::stats(content, 0);
    require(stats.accepted == 1,
            "replayed target replacement is committed but not counted as accepted draft");

    const auto * state = content.state(0);
    require(state->history_size == 4 && state->history[2].token == 4 &&
            state->history[3].token == 5,
            "replay commits every verified context token");
}

int main() {
    test_piece_classification_guards();
    test_math_evidence_requires_code_context();
    test_candidate_defaults();
    test_sidecar_capacity_reservations();
    test_relative_limits_and_override();
    test_exact_window_and_accepted_commit();
    test_maximum_32_token_window();
    test_fence_and_inline_code_modes();
    test_fence_tracking_uses_committed_prompt_only();
    test_request_replacement_and_sequence_isolation();
    test_provisional_is_separate();
    test_request_stats_reset_at_begin();
    test_stacked_verification_width_accounting();
    test_incremental_commit_matches_fresh_rebuild();
    test_context_limited_cycles_do_not_pollute_width_stats();
    test_replay_commit_and_draft_accounting_are_separate();
    std::puts("test-speculative-content: PASS");
    return 0;
}
