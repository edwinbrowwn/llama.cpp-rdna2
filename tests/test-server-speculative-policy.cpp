#include "server-speculative-policy.h"
#include "server-schema.h"

#include <cstdio>
#include <cstdlib>

namespace {

[[noreturn]] void fail(const char * message) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
}

void check(bool condition, const char * message) {
    if (!condition) {
        fail(message);
    }
}

server_speculative_draft_limit_input base_input() {
    return {
        15,
        -1,
        false,
        false,
    };
}

void test_default_and_request_cap() {
    auto input = base_input();
    check(server_speculative_draft_limit(input) == 15, "default draft limit changed");

    input.request_max = 4;
    check(server_speculative_draft_limit(input) == 4, "request cap ignored");

    input.request_max = 0;
    check(server_speculative_draft_limit(input) == 0, "zero request cap ignored");
}

void test_reasoning_pause() {
    auto input = base_input();
    input.reasoning_pause = true;
    check(server_speculative_draft_limit(input) == 15, "pause applied outside reasoning");

    input.reasoning_active = true;
    check(server_speculative_draft_limit(input) == 0, "active reasoning did not pause drafting");

    input.reasoning_pause = false;
    check(server_speculative_draft_limit(input) == 15, "reasoning paused without opt-in");
}

void test_request_schema() {
    common_params base;
    base.sampling.backend_sampling = true;

    const std::vector<llama_logit_bias> logit_bias_eog;
    const json enabled = {{"speculative.reasoning_pause", true}};
    auto params = server_schema::eval_llama_cmpl_schema(nullptr, base, logit_bias_eog, enabled);
    check(params.speculative.reasoning_pause, "request did not enable reasoning pause");
    check(!params.sampling.reasoning_tracking, "pause created a tracker without a speculator or boundaries");
    check(params.sampling.backend_sampling, "inert pause changed target backend sampling");

    base.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE};
    params = server_schema::eval_llama_cmpl_schema(nullptr, base, logit_bias_eog, enabled);
    check(!params.sampling.reasoning_tracking, "pause created a tracker without reasoning boundaries");
    check(params.sampling.backend_sampling, "boundary-less pause changed target backend sampling");

    base.sampling.reasoning_budget_start = {100};
    base.sampling.reasoning_budget_end = {{101}};
    params = server_schema::eval_llama_cmpl_schema(nullptr, base, logit_bias_eog, enabled);
    check(params.sampling.reasoning_tracking, "request did not enable token-level reasoning tracking");
    check(!params.sampling.backend_sampling, "active reasoning pause left target backend sampling enabled");

    const json capped_off = {
        {"speculative.n_max", 0},
        {"speculative.reasoning_pause", true},
    };
    params = server_schema::eval_llama_cmpl_schema(nullptr, base, logit_bias_eog, capped_off);
    check(!params.sampling.reasoning_tracking, "zero draft cap left reasoning tracking enabled");
    check(params.sampling.backend_sampling, "zero draft cap changed target backend sampling");

    base.speculative.reasoning_pause = true;
    const json disabled = {{"speculative.reasoning_pause", false}};
    params = server_schema::eval_llama_cmpl_schema(nullptr, base, logit_bias_eog, disabled);
    check(!params.speculative.reasoning_pause, "request did not override server-wide reasoning pause");
    check(!params.sampling.reasoning_tracking, "disabled pause left reasoning tracking enabled");
    check(params.sampling.backend_sampling, "disabled pause changed target backend sampling");
}

void test_pause_precedes_request_cap() {
    auto input = base_input();
    input.request_max = 8;
    input.reasoning_pause = true;
    input.reasoning_active = true;
    check(server_speculative_draft_limit(input) == 0, "request cap bypassed reasoning pause");

    input.reasoning_active = false;
    check(server_speculative_draft_limit(input) == 8, "request cap not restored after reasoning");
}

} // namespace

int main() {
    test_default_and_request_cap();
    test_reasoning_pause();
    test_request_schema();
    test_pause_precedes_request_cap();
    std::puts("server speculative reasoning-pause policy tests: PASS");
    return 0;
}
