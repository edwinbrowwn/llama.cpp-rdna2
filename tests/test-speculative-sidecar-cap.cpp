#include "speculative-sidecar-cap.h"
#include "ngram-map.h"

#include <cstdio>
#include <cstdlib>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "sidecar cap test failure: %s\n", message);
        std::abort();
    }
}

static void test_cap_and_explicit_override() {
    common_speculative_sidecar_cap_config cap { 3 };
    common_speculative_draft_params dp;
    dp.n_max = 8;
    require(common_speculative_sidecar_cap_request_enabled(cap, dp), "default request uses cap");
    require(common_speculative_sidecar_cap_limit(cap, dp) == 3, "cap follows configured neural width");

    require(common_speculative_neural_draft_limit(4, 5) == 4 &&
            common_speculative_neural_draft_limit(4, 3) == 3 &&
            common_speculative_neural_draft_limit(4, 0) == 0 &&
            common_speculative_neural_draft_limit(4, -1) == 4,
            "K4V verification envelope never widens neural generation");

    common_speculative_sidecar_cap_config content_cap { 7 };
    dp.n_max_ngram = 5;
    require(common_speculative_sidecar_cap_limit(content_cap, dp) == 5 &&
            common_speculative_proposal_limit(dp, true) == 5 &&
            common_speculative_proposal_limit(dp, false) == 8,
            "content-selected width applies only to the K4V proposal");
    dp.n_max_ngram = 4;
    require(common_speculative_sidecar_cap_limit(content_cap, dp) == 4,
            "runtime sidecar fallback narrows K4V to the configured baseline");
    dp.n_max = 3;
    dp.n_max_ngram = 5;
    require(common_speculative_sidecar_cap_limit(content_cap, dp) == 3,
            "remaining context is authoritative over K4V width");
    dp.n_max = 8;

    dp.n_max_user_override = true;
    require(!common_speculative_sidecar_cap_request_enabled(cap, dp), "explicit request bypasses cap");
}

static void test_ngram_map_fixed_width() {
    const llama_tokens prompt = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 1,
    };

    common_ngram_map normal(2, 6, true, 1);
    common_ngram_map_begin(normal, prompt);
    llama_tokens normal_draft;
    common_ngram_map_draft(normal, prompt, 2, normal_draft);
    require(normal_draft.size() == 6, "normal map uses configured width");
    common_ngram_map_accept(normal, 1);
    normal_draft.clear();
    common_ngram_map_draft(normal, prompt, 2, normal_draft);
    require(normal_draft.size() == 1, "normal map adapts to last accepted width");

    common_ngram_map capped(2, 6, true, 1);
    common_ngram_map_begin(capped, prompt);
    capped.draft_limit = 3;
    llama_tokens capped_draft;
    common_ngram_map_draft(capped, prompt, 2, capped_draft);
    require(capped_draft.size() == 3, "map honors fixed sidecar cap");
    common_ngram_map_accept(capped, 1);
    capped_draft.clear();
    common_ngram_map_draft(capped, prompt, 2, capped_draft);
    require(capped_draft.size() == 3, "fixed sidecar cap owns width after partial acceptance");

    common_ngram_map complex(2, 6, false, 1);
    common_ngram_map_begin(complex, prompt);
    complex.draft_limit = 3;
    llama_tokens complex_draft;
    common_ngram_map_draft(complex, prompt, 2, complex_draft);
    require(complex_draft.size() == 3, "complex map honors fixed sidecar cap");

    capped.draft_limit = 4;
    capped_draft.clear();
    common_ngram_map_draft(capped, prompt, 2, capped_draft);
    require(capped_draft.size() == 4,
            "content-selected K4V width can return to the baseline");
    capped.draft_limit = 5;
    capped_draft.clear();
    common_ngram_map_draft(capped, prompt, 2, capped_draft);
    require(capped_draft.size() == 5,
            "content-selected K4V width follows the wider cycle envelope");
    capped.draft_limit = 4;
    capped_draft.clear();
    common_ngram_map_draft(capped, prompt, 2, capped_draft);
    require(capped_draft.size() == 4,
            "content-selected K4V width narrows without stale continuation");

    common_ngram_map shifted(2, 6, true, 1);
    common_ngram_map_begin(shifted, prompt);
    llama_tokens shifted_draft;
    common_ngram_map_draft(shifted, prompt, 2, shifted_draft);
    const llama_tokens shorter(prompt.begin(), prompt.begin() + 12);
    shifted.draft_limit = 3;
    shifted_draft.clear();
    common_ngram_map_draft(shifted, shorter, 3, shifted_draft);
    require(shifted.idx_last_check == shorter.size() &&
            shifted.size_last_begin == shorter.size() &&
            shifted.draft_limit == 3,
            "K4V reconciles an in-generation context shrink without losing the selected cap");

    common_ngram_map short_value(2, 1, true, 1);
    common_ngram_map_begin(short_value, prompt);
    short_value.draft_limit = 6;
    llama_tokens short_draft;
    common_ngram_map_draft(short_value, prompt, 2, short_draft);
    require(short_draft.size() == 1, "cap preserves available value bound");
}

static void test_dynamic_mtp_deferred_widths() {
    require(common_speculative_sidecar_mtp_deferred_width_eligible(4, 0, 5),
            "fixed width four keeps five-row deferred catch-up");
    require(!common_speculative_sidecar_mtp_deferred_width_eligible(4, 0, 6),
            "fixed width four rejects an unreserved sixth row");
    require(common_speculative_sidecar_mtp_deferred_width_eligible(4, 1, 6),
            "K4V +1 keeps six-row deferred catch-up while MTP remains width four");
    require(common_speculative_sidecar_mtp_deferred_width_eligible(4, 3, 8),
            "K4V +3 keeps the complete eight-row target envelope");
    require(!common_speculative_sidecar_mtp_deferred_width_eligible(4, 3, 9),
            "deferred catch-up rejects rows beyond the content envelope");
    require(common_speculative_sidecar_mtp_deferred_width_eligible(5, 0, 5) &&
            common_speculative_sidecar_mtp_deferred_width_eligible(5, 0, 6),
            "existing width-five eligibility is preserved");
}

int main() {
    test_dynamic_mtp_deferred_widths();
    test_cap_and_explicit_override();
    test_ngram_map_fixed_width();
    std::puts("test-speculative-sidecar-cap: PASS");
    return 0;
}
