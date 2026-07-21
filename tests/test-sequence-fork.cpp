#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

struct batch_owner {
    llama_batch batch;

    explicit batch_owner(int32_t n_tokens) : batch(llama_batch_init(n_tokens, 0, 1)) {}
    ~batch_owner() { llama_batch_free(batch); }

    batch_owner(const batch_owner &) = delete;
    batch_owner & operator=(const batch_owner &) = delete;
};

static bool decode_tokens(
        llama_context * ctx,
        const llama_tokens & tokens,
        llama_pos pos_start,
        llama_seq_id seq_id,
        std::vector<float> * logits_out,
        int64_t * elapsed_us = nullptr) {
    if (tokens.empty()) {
        LOG_ERR("%s: token list is empty\n", __func__);
        return false;
    }

    const int64_t t_start = ggml_time_us();
    const size_t n_batch = llama_n_batch(ctx);
    for (size_t offset = 0; offset < tokens.size(); offset += n_batch) {
        const size_t count = std::min(n_batch, tokens.size() - offset);
        batch_owner owner((int32_t) count);
        for (size_t i = 0; i < count; ++i) {
            const bool output = offset + i + 1 == tokens.size();
            common_batch_add(owner.batch, tokens[offset + i],
                pos_start + (llama_pos) (offset + i), { seq_id }, output);
        }

        const int ret = llama_decode(ctx, owner.batch);
        if (ret != 0) {
            LOG_ERR("%s: llama_decode failed for seq %d at offset %zu with %d\n",
                __func__, seq_id, offset, ret);
            return false;
        }
    }

    const float * logits = llama_get_logits_ith(ctx, -1);
    if (logits == nullptr) {
        LOG_ERR("%s: missing logits for seq %d\n", __func__, seq_id);
        return false;
    }

    if (elapsed_us != nullptr) {
        *elapsed_us = ggml_time_us() - t_start;
    }

    if (logits_out != nullptr) {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
        logits_out->assign(logits, logits + n_vocab);
    }

    return true;
}

static bool get_seq_state(llama_context * ctx, llama_seq_id seq_id, std::vector<uint8_t> & state) {
    const size_t size = llama_state_seq_get_size(ctx, seq_id);
    if (size == 0) {
        LOG_ERR("%s: empty state for seq %d\n", __func__, seq_id);
        return false;
    }

    state.resize(size);
    const size_t copied = llama_state_seq_get_data(ctx, state.data(), state.size(), seq_id);
    if (copied != state.size()) {
        LOG_ERR("%s: copied %zu of %zu bytes for seq %d\n", __func__, copied, state.size(), seq_id);
        return false;
    }

    return true;
}

struct logit_diff {
    double max_abs = 0.0;
    double rms = 0.0;
    int max_index = -1;
    int argmax_a = -1;
    int argmax_b = -1;
};

static logit_diff compare_logits(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());

    logit_diff result;
    double sum_sq = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double diff = std::fabs((double) a[i] - (double) b[i]);
        sum_sq += diff*diff;
        if (diff > result.max_abs) {
            result.max_abs = diff;
            result.max_index = (int) i;
        }
    }
    result.rms = std::sqrt(sum_sq / std::max<size_t>(1, a.size()));
    result.argmax_a = std::max_element(a.begin(), a.end()) - a.begin();
    result.argmax_b = std::max_element(b.begin(), b.end()) - b.begin();
    return result;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.n_ctx = 2048;
    params.n_batch = 512;
    params.n_ubatch = 256;
    params.n_predict = 16; // number of fork cycles for this test
    params.sampling.seed = 1234;

    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    // Load the model once, then create the exact multi-sequence context needed
    // by the test. Avoid the common helper's default server/context lifecycle.
    auto model_init = common_init_from_params(params, true);
    llama_model * model = model_init->model();
    if (model == nullptr) {
        LOG_ERR("%s: failed to load model\n", __func__);
        return 1;
    }
    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        LOG_ERR("%s: model is not recurrent or hybrid\n", __func__);
        return 1;
    }

    auto cparams = common_context_params_to_llama(params);
    cparams.n_ctx = params.n_ctx;
    cparams.n_seq_max = 5;
    cparams.kv_unified = true;
    cparams.n_batch = std::max<uint32_t>(cparams.n_batch, 512);
    cparams.n_ubatch = std::min<uint32_t>(cparams.n_batch, std::max<uint32_t>(cparams.n_ubatch, 64));

    auto ctx = llama_context_ptr(llama_init_from_model(model, cparams));
    if (!ctx) {
        LOG_ERR("%s: failed to create context\n", __func__);
        return 1;
    }

    std::string prefix_text;
    if (!params.prompt.empty()) {
        prefix_text = params.prompt;
    } else {
        for (int i = 0; i < 32; ++i) {
            prefix_text += "The quick brown fox crosses the valley while the observatory records each step. ";
        }
    }
    const std::string suffix_text =
        "A different branch now studies the river, verifies the instruments, and records a deterministic result.";

    llama_tokens prefix;
    llama_tokens suffix;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        const int n_vocab = llama_vocab_n_tokens(vocab);
        for (int i = 0; i < 64; ++i) {
            prefix.push_back(1 + i % std::max(1, n_vocab - 1));
        }
        for (int i = 0; i < 8; ++i) {
            suffix.push_back(1 + (i + 67) % std::max(1, n_vocab - 1));
        }
    } else {
        prefix = common_tokenize(ctx.get(), prefix_text, true);
        suffix = common_tokenize(ctx.get(), suffix_text, false);
    }
    if (prefix.empty() || suffix.empty()) {
        LOG_ERR("%s: tokenization produced an empty prefix or suffix\n", __func__);
        return 1;
    }
    if (prefix.size() + suffix.size() >= llama_n_ctx_seq(ctx.get())) {
        LOG_ERR("%s: fixture requires %zu tokens but context is %u\n",
            __func__, prefix.size() + suffix.size(), llama_n_ctx_seq(ctx.get()));
        return 1;
    }

    LOG_INF("%s: model hybrid=%d recurrent=%d, n_ctx=%u, n_seq_max=%u, prefix=%zu, suffix=%zu, cycles=%d\n",
        __func__, (int) llama_model_is_hybrid(model), (int) llama_model_is_recurrent(model),
        llama_n_ctx(ctx.get()), llama_n_seq_max(ctx.get()), prefix.size(), suffix.size(), params.n_predict);

    constexpr llama_seq_id seq_source  = 0;
    constexpr llama_seq_id seq_fork_a  = 1;
    constexpr llama_seq_id seq_fork_b  = 2;
    constexpr llama_seq_id seq_clean_a = 3;
    constexpr llama_seq_id seq_clean_b = 4;

    // Build the canonical source boundary.
    if (!decode_tokens(ctx.get(), prefix, 0, seq_source, nullptr)) {
        return 1;
    }

    std::vector<uint8_t> source_state_before;
    if (!get_seq_state(ctx.get(), seq_source, source_state_before)) {
        return 1;
    }

    // Build two clean, independently evaluated reference sequences. Their spread
    // establishes the backend/layout reproducibility floor before seq_cp is used.
    std::vector<float> logits_clean_a;
    std::vector<float> logits_clean_a_repeat;
    std::vector<float> logits_clean_b;
    int64_t clean_decode_us = 0;
    int64_t clean_repeat_decode_us = 0;
    int64_t clean_b_decode_us = 0;
    if (!decode_tokens(ctx.get(), prefix, 0, seq_clean_a, nullptr) ||
        !decode_tokens(ctx.get(), suffix, (llama_pos) prefix.size(), seq_clean_a, &logits_clean_a, &clean_decode_us) ||
        !llama_memory_seq_rm(llama_get_memory(ctx.get()), seq_clean_a, -1, -1) ||
        !decode_tokens(ctx.get(), prefix, 0, seq_clean_a, nullptr) ||
        !decode_tokens(ctx.get(), suffix, (llama_pos) prefix.size(), seq_clean_a, &logits_clean_a_repeat, &clean_repeat_decode_us) ||
        !decode_tokens(ctx.get(), prefix, 0, seq_clean_b, nullptr) ||
        !decode_tokens(ctx.get(), suffix, (llama_pos) prefix.size(), seq_clean_b, &logits_clean_b, &clean_b_decode_us)) {
        return 1;
    }
    const logit_diff diff_clean_same = compare_logits(logits_clean_a, logits_clean_a_repeat);
    const logit_diff diff_clean = compare_logits(logits_clean_a, logits_clean_b);
    LOG_INF("%s: clean same-seq repeat: max_abs=%g rms=%g argmax=%d/%d\n",
        __func__, diff_clean_same.max_abs, diff_clean_same.rms,
        diff_clean_same.argmax_a, diff_clean_same.argmax_b);
    LOG_INF("%s: clean A/B: max_abs=%g rms=%g argmax=%d/%d\n",
        __func__, diff_clean.max_abs, diff_clean.rms, diff_clean.argmax_a, diff_clean.argmax_b);

    if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), seq_clean_a, -1, -1) ||
        !llama_memory_seq_rm(llama_get_memory(ctx.get()), seq_clean_b, -1, -1)) {
        LOG_ERR("%s: failed to remove clean reference sequences\n", __func__);
        return 1;
    }

    const std::vector<float> & logits_reference = logits_clean_a;

    const int cycles = std::max(1, params.n_predict);
    std::vector<int64_t> fork_us;
    std::vector<int64_t> fork_decode_us;
    fork_us.reserve(cycles);
    fork_decode_us.reserve(cycles);

    constexpr double eps = 1e-4;
    const double allowed_max_abs = std::max(eps, 1.5*diff_clean.max_abs);
    const double allowed_rms     = std::max(eps, 1.5*diff_clean.rms);
    for (int cycle = 0; cycle < cycles; ++cycle) {
        const int64_t t_fork = ggml_time_us();
        llama_memory_seq_cp(llama_get_memory(ctx.get()), seq_source, seq_fork_a, -1, -1);
        llama_memory_seq_cp(llama_get_memory(ctx.get()), seq_source, seq_fork_b, -1, -1);
        fork_us.push_back(ggml_time_us() - t_fork);

        if (cycle == 0) {
            std::vector<uint8_t> state_source_shared;
            std::vector<uint8_t> state_fork_shared;
            if (!get_seq_state(ctx.get(), seq_source, state_source_shared) ||
                !get_seq_state(ctx.get(), seq_fork_a, state_fork_shared)) {
                return 1;
            }
            const size_t state_header_size = sizeof(uint32_t) + sizeof(llama_seq_id);
            const bool same_state = state_source_shared.size() == state_fork_shared.size() &&
                state_source_shared.size() >= state_header_size &&
                std::equal(state_source_shared.begin() + state_header_size, state_source_shared.end(),
                           state_fork_shared.begin() + state_header_size);
            if (!same_state) {
                const size_t n = std::min(state_source_shared.size(), state_fork_shared.size());
                size_t first_diff = state_header_size;
                while (first_diff < n && state_source_shared[first_diff] == state_fork_shared[first_diff]) {
                    ++first_diff;
                }
                LOG_WRN("%s: source/fork serialized layouts differ after seq_cp as expected for different physical cells: sizes=%zu/%zu first_diff=%zu\n",
                    __func__, state_source_shared.size(), state_fork_shared.size(), first_diff);
            } else {
                LOG_INF("%s: source/fork serialized states match immediately after seq_cp (%zu bytes)\n",
                    __func__, state_source_shared.size());
            }
        }

        std::vector<float> logits_fork_a;
        std::vector<float> logits_fork_b;
        int64_t elapsed_a_us = 0;
        int64_t elapsed_b_us = 0;
        if (!decode_tokens(ctx.get(), suffix, (llama_pos) prefix.size(), seq_fork_a, &logits_fork_a, &elapsed_a_us) ||
            !decode_tokens(ctx.get(), suffix, (llama_pos) prefix.size(), seq_fork_b, &logits_fork_b, &elapsed_b_us)) {
            LOG_ERR("%s: fork decode failed at cycle %d\n", __func__, cycle);
            return 1;
        }
        fork_decode_us.push_back(elapsed_a_us);
        fork_decode_us.push_back(elapsed_b_us);

        const logit_diff diff_ab = compare_logits(logits_fork_a, logits_fork_b);
        const logit_diff diff_ar = compare_logits(logits_reference, logits_fork_a);
        const logit_diff diff_br = compare_logits(logits_reference, logits_fork_b);
        if (diff_clean_same.max_abs > allowed_max_abs || diff_clean_same.rms > allowed_rms ||
            diff_clean_same.argmax_a != diff_clean_same.argmax_b ||
            diff_ab.max_abs > allowed_max_abs || diff_ab.rms > allowed_rms || diff_ab.argmax_a != diff_ab.argmax_b ||
            diff_ar.max_abs > allowed_max_abs || diff_ar.rms > allowed_rms || diff_ar.argmax_a != diff_ar.argmax_b ||
            diff_br.max_abs > allowed_max_abs || diff_br.rms > allowed_rms || diff_br.argmax_a != diff_br.argmax_b) {
            LOG_ERR("%s: cycle %d fork A/B: max_abs=%g rms=%g argmax=%d/%d\n",
                __func__, cycle, diff_ab.max_abs, diff_ab.rms, diff_ab.argmax_a, diff_ab.argmax_b);
            LOG_ERR("%s: cycle %d clean/A: max_abs=%g rms=%g argmax=%d/%d\n",
                __func__, cycle, diff_ar.max_abs, diff_ar.rms, diff_ar.argmax_a, diff_ar.argmax_b);
            LOG_ERR("%s: cycle %d clean/B: max_abs=%g rms=%g argmax=%d/%d\n",
                __func__, cycle, diff_br.max_abs, diff_br.rms, diff_br.argmax_a, diff_br.argmax_b);
            return 1;
        }

        if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), seq_fork_a, -1, -1) ||
            !llama_memory_seq_rm(llama_get_memory(ctx.get()), seq_fork_b, -1, -1)) {
            LOG_ERR("%s: failed to remove fork sequences at cycle %d\n", __func__, cycle);
            return 1;
        }
    }

    std::vector<uint8_t> source_state_after;
    if (!get_seq_state(ctx.get(), seq_source, source_state_after)) {
        return 1;
    }
    if (source_state_before != source_state_after) {
        LOG_ERR("%s: source sequence state changed after fork cycles (%zu vs %zu bytes)\n",
            __func__, source_state_before.size(), source_state_after.size());
        return 1;
    }

    const auto average = [](const std::vector<int64_t> & values) {
        return values.empty() ? 0.0 :
            (double) std::accumulate(values.begin(), values.end(), int64_t(0)) / values.size();
    };
    const auto maximum = [](const std::vector<int64_t> & values) {
        return values.empty() ? int64_t(0) : *std::max_element(values.begin(), values.end());
    };

    LOG_INF("%s: PASS\n", __func__);
    LOG_INF("%s: fork metadata avg %.3f ms, max %.3f ms\n",
        __func__, average(fork_us)/1000.0, maximum(fork_us)/1000.0);
    LOG_INF("%s: clean suffix %.3f/%.3f/%.3f ms; fork suffix avg %.3f ms, max %.3f ms\n",
        __func__, clean_decode_us/1000.0, clean_repeat_decode_us/1000.0, clean_b_decode_us/1000.0,
        average(fork_decode_us)/1000.0, maximum(fork_decode_us)/1000.0);
    LOG_INF("%s: variability allowance max_abs=%g rms=%g (1.5x independent clean A/B)\n",
        __func__, allowed_max_abs, allowed_rms);
    LOG_INF("%s: source state preserved (%zu bytes), argmax token %d\n",
        __func__, source_state_after.size(), compare_logits(logits_reference, logits_reference).argmax_a);

    return 0;
}