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

struct common_speculative_content_test_access {
    static void configure(common_speculative_content & content, int base_nmax,
            int max_boost, int window, bool provisional = false, bool trace = false) {
        content.config_ = {};
        content.config_.enabled = max_boost > 0;
        content.config_.max_boost = max_boost;
        content.config_.window = window;
        content.config_.hysteresis = 2;
        content.config_.provisional = provisional;
        content.config_.trace = trace;
        content.base_nmax_ = base_nmax;
        content.token_flags_.assign(32, STF_ALPHA);
        content.token_flags_[5] = STF_BRACE | STF_STRONG_ANCHOR;
        content.states_.assign(1, {});
    }

    static const common_speculative_content_stats & stats(
            const common_speculative_content & content, int bucket) {
        return content.stats_[bucket];
    }
};

static void test_sidecar_capacity_reservations() {
    set_test_env("SPEC_CONTENT_BOOST", "1");
    set_test_env("SPEC_CONTENT_MAX_BOOST", "3");

    common_params params;
    params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    params.speculative.draft.n_max = 4;
    params.speculative.draft.sidecar_candidate_ready = true;

    require(common_speculative_n_max(&params.speculative) == 7,
            "sidecar candidate reserves target outputs through base plus boost");
    require(common_context_params_to_llama(params).n_rs_seq == 7,
            "sidecar candidate reserves recurrent rollback through base plus boost");

    params.speculative.synth_len = 1.0;
    require(common_speculative_n_max(&params.speculative) == 4,
            "synthetic mode does not acquire content output capacity");
    require(common_context_params_to_llama(params).n_rs_seq == 4,
            "synthetic mode does not acquire content rollback capacity");

    params.speculative.synth_len = -1.0;
    params.speculative.draft.sidecar_candidate_ready = false;
    require(common_speculative_n_max(&params.speculative) == 4,
            "native draft path does not acquire content output capacity");
    require(common_context_params_to_llama(params).n_rs_seq == 4,
            "native draft path does not acquire content rollback capacity");
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
    require(content.max_limit(4, 128, false) == 7,
            "maximum envelope is baseline plus configured boost");

    require(content.selected_limit(4, 6, true, SPEC_BOOST_0) == 6,
            "explicit request above baseline remains exact");
    require(content.selected_limit(4, 2, true, SPEC_BOOST_3) == 2,
            "explicit request below baseline remains exact");
    require(content.max_limit(4, 6, true) == 6,
            "explicit request bypasses automatic maximum");
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
    content.observe_draft(0, draft, 2, 4, 4, 7);
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

static void test_provisional_is_separate() {
    common_speculative_content content;
    common_speculative_content_test_access::configure(content, 4, 3, 4, true);

    const llama_token prompt[] = { 0, 1, 2 };
    content.prepare(0, prompt, 3, 3);
    const llama_token draft[] = { 5, 6 };
    content.observe_draft(0, draft, 2, 4, 4, 7);

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

static void test_replay_commit_and_draft_accounting_are_separate() {
    common_speculative_content content;
    common_speculative_content_test_access::configure(content, 4, 3, 4);

    const llama_token prompt[] = { 0, 1, 2 };
    content.prepare(0, prompt, 3, 3);
    const llama_token replay[] = { 4, 5 };
    content.observe_draft(0, replay, 2, 4, 4, 7);
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
    test_sidecar_capacity_reservations();
    test_relative_limits_and_override();
    test_exact_window_and_accepted_commit();
    test_maximum_32_token_window();
    test_provisional_is_separate();
    test_replay_commit_and_draft_accounting_are_separate();
    std::puts("test-speculative-content: PASS");
    return 0;
}
