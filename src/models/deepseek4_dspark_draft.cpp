#include "models.h"

#include "llama-model-loader.h"

#include <algorithm>
#include <stdexcept>

void llama_model_deepseek4_dspark_draft::load_hparams(llama_model_loader & ml) {
    // This drafter artifact intentionally contains only its recipe and weights;
    // it is loaded alongside the target model, which owns the tokenizer/vocab.
    const gguf_context * ctx = ml.metadata;
    for (int i = 0; i < gguf_get_n_kv(ctx); ++i) {
        if (gguf_get_kv_type(ctx, i) == GGUF_TYPE_ARRAY) {
            continue;
        }
        gguf_kv.emplace(gguf_get_key(ctx, i), gguf_kv_to_str(ctx, i));
    }
    ml.get_key(LLM_KV_GENERAL_NAME, name, false);

    uint32_t layer_count = 0;
    uint32_t block_size = 0;
    uint32_t markov_rank = 0;
    if (!ml.get_key("dspark.layer_count", layer_count) || layer_count == 0 ||
        !ml.get_key("dspark.block_size", block_size) || block_size == 0 ||
        !ml.get_key("dspark.markov_rank", markov_rank) || markov_rank == 0) {
        throw std::runtime_error("DeepSeek-V4 DSpark drafter requires dspark.layer_count, dspark.block_size, and dspark.markov_rank");
    }
    if (!ml.get_arr(LLM_KV_DSPARK_TARGET_LAYER_IDS, target_layer_ids) || target_layer_ids.size() != layer_count) {
        throw std::runtime_error("DeepSeek-V4 DSpark drafter requires dspark.target_layer_ids matching dspark.layer_count");
    }

    hparams.n_ctx_train = 1;
    hparams.n_embd = 4096;
    hparams.n_layer_all = layer_count;
    hparams.n_expert = 256;
    hparams.n_expert_used = 8;
    hparams.n_embd_head_k_full = 512;
    hparams.n_embd_head_v_full = 512;
    hparams.n_embd_head_k_swa = 512;
    hparams.n_embd_head_v_swa = 512;
    hparams.n_rot_full = 64;
    hparams.n_rot_swa = 64;
    hparams.n_lora_q = 1024;
    hparams.n_ff_exp = 2048;
    hparams.n_expert_shared = 1;
    hparams.dsv4_o_group_count = 8;
    hparams.dsv4_o_lora_rank = 1024;
    hparams.dsv4_hc_mult = 4;
    hparams.f_norm_rms_eps = 1e-5f;
    hparams.rope_freq_base_train = 10000.0f;
    hparams.rope_freq_scale_train = 1.0f;
    hparams.rope_type = LLAMA_ROPE_TYPE_NEOX;
    hparams.causal_attn = true;
    hparams.swa_type = LLAMA_SWA_TYPE_NONE;
    std::fill(hparams.n_head_arr.begin(), hparams.n_head_arr.end(), 64);
    std::fill(hparams.n_head_kv_arr.begin(), hparams.n_head_kv_arr.end(), 1);
    std::fill(hparams.n_ff_arr.begin(), hparams.n_ff_arr.end(), 2048);

    type = LLM_TYPE_UNKNOWN;
    LLAMA_LOG_INFO("%s: DeepSeek-V4 DSpark drafter: layers=%u, target_layers=[%d,%d,%d], block_size=%u, markov_rank=%u\n",
        __func__, layer_count, target_layer_ids[0], target_layer_ids[1], target_layer_ids[2], block_size, markov_rank);
}

void llama_model_deepseek4_dspark_draft::load_vocab(llama_model_loader &) {
    // No tokenizer is serialized in the official drafter artifact.  The target
    // model supplies vocabulary semantics when the drafter is used.
}

void llama_model_deepseek4_dspark_draft::load_arch_hparams(llama_model_loader &) {
    // All recipe values are loaded above because the artifact does not carry the
    // normal transformer metadata namespace.
}

void llama_model_deepseek4_dspark_draft::load_arch_tensors(llama_model_loader & ml) {
    const int n_layer = hparams.n_layer();
    const int64_t n_embd = hparams.n_embd;
    const int64_t n_ff = hparams.n_ff_exp;
    const int64_t n_expert = hparams.n_expert;
    const int64_t hc_dim = hparams.dsv4_hc_mult * n_embd;
    const int64_t hc_mix_dim = (2 + hparams.dsv4_hc_mult) * hparams.dsv4_hc_mult;

    const ggml_tensor * markov_meta = ml.get_tensor_meta("dspark.markov_w1.weight");
    if (markov_meta == nullptr) {
        throw std::runtime_error("DeepSeek-V4 DSpark drafter is missing dspark.markov_w1.weight");
    }
    const int64_t vocab_size = markov_meta->ne[1];
    const int64_t markov_rank = markov_meta->ne[0];
    if (markov_rank != 256) {
        throw std::runtime_error("DeepSeek-V4 DSpark drafter has unsupported markov rank");
    }

    fc = create_tensor(tn(LLM_TENSOR_DSPARK_MAIN_PROJ, "weight"), {3 * n_embd, n_embd}, 0);
    output_norm_enc = create_tensor(tn(LLM_TENSOR_DSPARK_MAIN_NORM, "weight"), {n_embd}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_DSPARK_NORM, "weight"), {n_embd}, 0);

    dspark_markov_w1 = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W1_OFFICIAL, "weight"), {markov_rank, vocab_size}, 0);
    dspark_markov_w2 = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W2_OFFICIAL, "weight"), {markov_rank, vocab_size}, 0);
    dspark_conf_proj = create_tensor(tn(LLM_TENSOR_DSPARK_CONFIDENCE_HEAD, "weight"), {n_embd + markov_rank}, 0);
    dspark_conf_proj_b = nullptr;

    hc_head_fn = create_tensor(tn(LLM_TENSOR_DSPARK_HC_HEAD_FN_OFFICIAL, "weight"), {hc_dim, hparams.dsv4_hc_mult}, 0);
    hc_head_base = create_tensor(tn(LLM_TENSOR_DSPARK_HC_HEAD_BASE_OFFICIAL, "weight"), {hparams.dsv4_hc_mult}, 0);
    hc_head_scale = create_tensor(tn(LLM_TENSOR_DSPARK_HC_HEAD_SCALE_OFFICIAL, "weight"), {1}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];
        layer.attn_norm = create_tensor(tn(LLM_TENSOR_DSPARK_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.attn_sinks = create_tensor(tn(LLM_TENSOR_DSPARK_ATTN_SINKS, "weight", i), {64}, 0);
        layer.wq_a = create_tensor(tn(LLM_TENSOR_DSPARK_ATTN_Q_A, "weight", i), {n_embd, hparams.n_lora_q}, 0);
        layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_DSPARK_ATTN_Q_A_NORM, "weight", i), {hparams.n_lora_q}, 0);
        layer.wq_b = create_tensor(tn(LLM_TENSOR_DSPARK_ATTN_Q_B, "weight", i), {hparams.n_lora_q, 64 * 512}, 0);
        layer.wkv = create_tensor(tn(LLM_TENSOR_DSPARK_ATTN_KV, "weight", i), {n_embd, 512}, 0);
        layer.attn_kv_norm = create_tensor(tn(LLM_TENSOR_DSPARK_ATTN_KV_A_NORM, "weight", i), {512}, 0);
        layer.wo_a = create_tensor(tn(LLM_TENSOR_DSPARK_ATTN_OUT_A, "weight", i), {4096, 8192}, 0);
        layer.wo_b = create_tensor(tn(LLM_TENSOR_DSPARK_ATTN_OUT_B, "weight", i), {8192, n_embd}, 0);

        layer.hc_attn_fn = create_tensor(tn(LLM_TENSOR_DSPARK_HC_ATTN_FN, "weight", i), {hc_dim, hc_mix_dim}, 0);
        layer.hc_attn_base = create_tensor(tn(LLM_TENSOR_DSPARK_HC_ATTN_BASE, "weight", i), {hc_mix_dim}, 0);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_DSPARK_HC_ATTN_SCALE, "weight", i), {3}, 0);
        layer.hc_ffn_fn = create_tensor(tn(LLM_TENSOR_DSPARK_HC_FFN_FN, "weight", i), {hc_dim, hc_mix_dim}, 0);
        layer.hc_ffn_base = create_tensor(tn(LLM_TENSOR_DSPARK_HC_FFN_BASE, "weight", i), {hc_mix_dim}, 0);
        layer.hc_ffn_scale = create_tensor(tn(LLM_TENSOR_DSPARK_HC_FFN_SCALE, "weight", i), {3}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_DSPARK_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_DSPARK_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_DSPARK_EXP_PROBS_B, "bias", i), {n_expert}, 0);
        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_DSPARK_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff, n_expert}, 0);
        layer.ffn_up_exps = create_tensor(tn(LLM_TENSOR_DSPARK_FFN_UP_EXPS, "weight", i), {n_embd, n_ff, n_expert}, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_DSPARK_FFN_DOWN_EXPS, "weight", i), {n_ff, n_embd, n_expert}, 0);
        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_DSPARK_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_up_shexp = create_tensor(tn(LLM_TENSOR_DSPARK_FFN_UP_SHEXP, "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_DSPARK_FFN_DOWN_SHEXP, "weight", i), {n_ff, n_embd}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_deepseek4_dspark_draft::build_arch_graph(const llm_graph_params &) const {
    throw std::runtime_error("DeepSeek-V4 DSpark drafter graph is not implemented: loader boundary reached after loading dspark.* tensors");
}