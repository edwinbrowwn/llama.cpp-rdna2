#include "arg.h"
#include "common.h"
#include "log.h"
#include "speculative.h"

#include <algorithm>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

struct batch_owner {
    llama_batch batch;
    explicit batch_owner(int32_t n_tokens) : batch(llama_batch_init(n_tokens, 0, 1)) {}
    ~batch_owner() { llama_batch_free(batch); }
    batch_owner(const batch_owner &) = delete;
    batch_owner & operator=(const batch_owner &) = delete;
};

static bool decode_prompt(
        llama_context * ctx_tgt,
        common_speculative * spec,
        const llama_tokens & tokens,
        llama_seq_id seq_id,
        llama_pos pos_start = 0) {
    const size_t n_batch = llama_n_batch(ctx_tgt);
    for (size_t offset = 0; offset < tokens.size(); offset += n_batch) {
        const size_t count = std::min(n_batch, tokens.size() - offset);
        batch_owner owner((int32_t) count);
        for (size_t i = 0; i < count; ++i) {
            // MTP process() consumes every target next-n embedding row.
            common_batch_add(owner.batch, tokens[offset + i], pos_start + (llama_pos) (offset + i), { seq_id }, true);
        }
        if (llama_decode(ctx_tgt, owner.batch) != 0) {
            LOG_ERR("%s: target decode failed at offset %zu\n", __func__, offset);
            return false;
        }
        if (!common_speculative_process(spec, owner.batch)) {
            LOG_ERR("%s: MTP process failed at offset %zu\n", __func__, offset);
            return false;
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.n_ctx = 512;
    params.n_batch = 256;
    params.n_ubatch = 128;
    params.n_parallel = 3;
    params.sampling.seed = 1234;
    params.n_predict = 32; // repeated target/draft shadow-restore cycles

    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    params.n_parallel = 3;
    params.kv_unified = true;
    params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    params.speculative.draft.n_max = 3;
    params.speculative.draft.n_min = 0;
    params.speculative.draft.p_min = 0.0f;
    params.speculative.draft.backend_sampling = false;

    ggml_backend_load_all();

    auto model_init = common_init_from_params(params, true);
    llama_model * model = model_init->model();
    if (model == nullptr) {
        LOG_ERR("%s: failed to load model\n", __func__);
        return 1;
    }
    if (!llama_model_is_hybrid(model) && !llama_model_is_recurrent(model)) {
        LOG_ERR("%s: model is not hybrid/recurrent\n", __func__);
        return 1;
    }

    auto cparams = common_context_params_to_llama(params);
    cparams.n_ctx = params.n_ctx;
    cparams.n_seq_max = 3;
    cparams.kv_unified = true;
    auto ctx_tgt = llama_context_ptr(llama_init_from_model(model, cparams));
    if (!ctx_tgt) {
        LOG_ERR("%s: failed to create target context\n", __func__);
        return 1;
    }

    auto spec_init = common_speculative_init_from_params(params, model, ctx_tgt.get());
    llama_context * ctx_dft = spec_init ? spec_init->context() : nullptr;
    if (ctx_dft == nullptr) {
        LOG_ERR("%s: failed to create MTP draft context\n", __func__);
        return 1;
    }

    params.speculative.draft.ctx_tgt = ctx_tgt.get();
    params.speculative.draft.ctx_dft = ctx_dft;
    common_speculative_ptr spec(common_speculative_init(params.speculative, 3));
    if (!spec) {
        LOG_ERR("%s: failed to initialize MTP driver\n", __func__);
        return 1;
    }

    llama_tokens prompt;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        const int n_vocab = llama_vocab_n_tokens(vocab);
        for (int i = 0; i < 64; ++i) {
            prompt.push_back(1 + i % std::max(1, n_vocab - 1));
        }
    } else {
        std::string text = params.prompt;
        if (text.empty()) {
            for (int i = 0; i < 16; ++i) {
                text += "A deterministic MTP boundary records every observation before the sequence is forked. ";
            }
        }
        prompt = common_tokenize(ctx_tgt.get(), text, true);
    }
    if (prompt.empty()) {
        LOG_ERR("%s: empty prompt\n", __func__);
        return 1;
    }

    constexpr llama_seq_id seq_source = 0;
    constexpr llama_seq_id seq_copied = 1;
    constexpr llama_seq_id seq_missing = 2;

    if (!decode_prompt(ctx_tgt.get(), spec.get(), prompt, seq_source)) {
        return 1;
    }

    const llama_pos tgt_pos = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt.get()), seq_source);
    const llama_pos dft_pos = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_source);
    if (tgt_pos != (llama_pos) prompt.size() - 1 || dft_pos != tgt_pos) {
        LOG_ERR("%s: target/draft positions disagree: target=%d draft=%d expected=%zu\n",
            __func__, tgt_pos, dft_pos, prompt.size() - 1);
        return 1;
    }

    std::vector<uint8_t> state_source;
    if (!common_speculative_get_state(spec.get(), seq_source, state_source) || state_source.empty()) {
        LOG_ERR("%s: MTP source pending state is unavailable\n", __func__);
        return 1;
    }

    llama_memory_seq_cp(llama_get_memory(ctx_tgt.get()), seq_source, seq_copied, -1, -1);
    llama_memory_seq_cp(llama_get_memory(ctx_tgt.get()), seq_source, seq_missing, -1, -1);
    llama_memory_seq_cp(llama_get_memory(ctx_dft), seq_source, seq_copied, -1, -1);
    llama_memory_seq_cp(llama_get_memory(ctx_dft), seq_source, seq_missing, -1, -1);

    common_speculative_set_state(spec.get(), seq_copied, state_source);

    std::vector<uint8_t> state_copied;
    std::vector<uint8_t> state_missing;
    if (!common_speculative_get_state(spec.get(), seq_copied, state_copied) ||
        !common_speculative_get_state(spec.get(), seq_missing, state_missing)) {
        LOG_ERR("%s: failed to read forked MTP state\n", __func__);
        return 1;
    }
    if (state_source != state_copied) {
        LOG_ERR("%s: copied MTP pending state differs (%zu/%zu bytes)\n",
            __func__, state_source.size(), state_copied.size());
        return 1;
    }
    if (state_source == state_missing) {
        LOG_ERR("%s: negative-control MTP state unexpectedly matches source\n", __func__);
        return 1;
    }

    llama_tokens result_copied;
    llama_tokens result_missing;
    llama_tokens * results[] = { nullptr, &result_copied, &result_missing };
    for (llama_seq_id seq_id : { seq_copied, seq_missing }) {
        auto & dp = common_speculative_get_draft_params(spec.get(), seq_id);
        dp.drafting = true;
        dp.n_max = 3;
        dp.n_past = (llama_pos) prompt.size();
        dp.id_last = prompt.back();
        dp.prompt = &prompt;
        dp.result = results[seq_id];
        common_speculative_begin(spec.get(), seq_id, prompt);
    }
    common_speculative_draft(spec.get());

    if (result_copied.empty()) {
        LOG_ERR("%s: copied fork produced no baseline draft\n", __func__);
        return 1;
    }
    const bool missing_control_equal = result_copied == result_missing;

    // Reset temporary controls. The source sequence remains the immutable shadow.
    llama_memory_seq_rm(llama_get_memory(ctx_tgt.get()), seq_copied, -1, -1);
    llama_memory_seq_rm(llama_get_memory(ctx_tgt.get()), seq_missing, -1, -1);
    llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_copied, -1, -1);
    llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_missing, -1, -1);

    llama_tokens suffix;
    const char * suffix_env = getenv("GGML_TEST_FORK_SUFFIX_TOKENS");
    const int requested_suffix_tokens = suffix_env ? std::max(1, atoi(suffix_env)) : 0;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        const int n_vocab = llama_vocab_n_tokens(vocab);
        const int n_suffix = requested_suffix_tokens > 0 ? requested_suffix_tokens : 8;
        for (int i = 0; i < n_suffix; ++i) {
            suffix.push_back(1 + (i + 83) % std::max(1, n_vocab - 1));
        }
    } else if (requested_suffix_tokens > 0) {
        std::string suffix_text;
        for (int i = 0; i < requested_suffix_tokens; ++i) {
            suffix_text += "token ";
        }
        suffix = common_tokenize(ctx_tgt.get(), suffix_text, false);
    } else {
        suffix = common_tokenize(ctx_tgt.get(),
            "Continue the forked MTP branch with a deterministic short observation.", false);
    }
    llama_tokens prompt_with_suffix = prompt;
    prompt_with_suffix.insert(prompt_with_suffix.end(), suffix.begin(), suffix.end());

    llama_tokens active_tail;
    const char * tail_env = getenv("GGML_TEST_FORK_ACTIVE_TAIL_TOKENS");
    const int requested_tail_tokens = tail_env ? std::max(0, atoi(tail_env)) : 0;
    if (requested_tail_tokens > 0) {
        const int n_vocab = llama_vocab_n_tokens(vocab);
        for (int i = 0; i < requested_tail_tokens; ++i) {
            active_tail.push_back(1 + (i + 101) % std::max(1, n_vocab - 1));
        }
    }

    llama_tokens draft_reference;
    const int cycles = std::max(1, params.n_predict);
    for (int cycle = 0; cycle < cycles; ++cycle) {
        llama_memory_seq_cp(llama_get_memory(ctx_tgt.get()), seq_source, seq_copied, -1, -1);
        llama_memory_seq_cp(llama_get_memory(ctx_dft), seq_source, seq_copied, -1, -1);
        common_speculative_set_state(spec.get(), seq_copied, state_source);

        if (!decode_prompt(ctx_tgt.get(), spec.get(), suffix, seq_copied, (llama_pos) prompt.size())) {
            LOG_ERR("%s: target/MTP suffix processing failed at cycle %d\n", __func__, cycle);
            return 1;
        }

        llama_tokens draft;
        auto & dp = common_speculative_get_draft_params(spec.get(), seq_copied);
        dp.drafting = true;
        dp.n_max = 3;
        dp.n_past = (llama_pos) prompt_with_suffix.size();
        dp.id_last = prompt_with_suffix.back();
        dp.prompt = &prompt_with_suffix;
        dp.result = &draft;
        common_speculative_begin(spec.get(), seq_copied, prompt_with_suffix);
        common_speculative_draft(spec.get());

        if (cycle == 0) {
            draft_reference = draft;
        } else if (draft != draft_reference) {
            LOG_ERR("%s: repeated MTP draft differs at cycle %d (reference=%zu current=%zu)\n",
                __func__, cycle, draft_reference.size(), draft.size());
            return 1;
        }

        if (!active_tail.empty()) {
            // Draft generation has populated speculative positions in ctx_dft.
            // The server discards that unaccepted region before processing the
            // real continuation.
            if (!llama_memory_seq_rm(
                    llama_get_memory(ctx_dft), seq_copied,
                    (llama_pos) prompt_with_suffix.size(), -1)) {
                LOG_ERR("%s: failed to remove speculative draft region at cycle %d\n", __func__, cycle);
                return 1;
            }
        }
        if (!active_tail.empty() && !decode_prompt(
                ctx_tgt.get(), spec.get(), active_tail, seq_copied,
                (llama_pos) prompt_with_suffix.size())) {
            LOG_ERR("%s: active continuation tail failed at cycle %d\n", __func__, cycle);
            return 1;
        }

        llama_memory_seq_rm(llama_get_memory(ctx_tgt.get()), seq_copied, -1, -1);
        llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_copied, -1, -1);
    }

    LOG_INF("%s: PASS model hybrid=%d prompt=%zu suffix=%zu active_tail=%zu state=%zu bytes draft_tokens=%zu missing_control_equal=%d cycles=%d\n",
        __func__, (int) llama_model_is_hybrid(model), prompt.size(), suffix.size(), active_tail.size(), state_source.size(),
        draft_reference.size(), (int) missing_control_equal, cycles);
    return 0;
}