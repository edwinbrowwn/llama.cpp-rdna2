#include "models.h"

#include "llama-model-loader.h"

#include <algorithm>
#include <cmath>
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
    if (layer_count != 3 || !ml.get_arr(LLM_KV_DSPARK_TARGET_LAYER_IDS, target_layer_ids) ||
            target_layer_ids.size() != layer_count ||
            target_layer_ids[0] != 40 || target_layer_ids[1] != 41 || target_layer_ids[2] != 42) {
        throw std::runtime_error("DeepSeek-V4 DSpark drafter requires target layers [40,41,42]");
    }

    // The official artifact has no embedding vocabulary.  Its encoder input is
    // the concatenation of the three hidden states captured from the target.
    hparams.n_ctx_train = 1;
    hparams.n_embd = 4096;
    hparams.n_embd_inp_enc_impl = layer_count * hparams.n_embd;
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
    hparams.dsv4_hc_sinkhorn_iters = 4;
    hparams.dsv4_hc_eps = 1e-6f;
    hparams.expert_weights_norm = true;
    hparams.expert_weights_scale = 1.0f;
    hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS;
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

std::unique_ptr<llm_graph_context> llama_model_deepseek4_dspark_draft::build_arch_graph(const llm_graph_params & params) const {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            return std::make_unique<graph<true>>(*this, params);
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            return std::make_unique<graph<false>>(*this, params);
        default:
            GGML_ABORT("invalid DeepSeek-V4 DSpark graph type");
    }
}

template <>
ggml_tensor * llama_model_deepseek4_dspark_draft::graph<true>::build_inp_embd_enc() const {
    // The caller supplies target hidden states in target-layer order [40,41,42],
    // matching the existing DFlash/EAGLE encoder input convention.
    auto inp_target = std::make_unique<llm_graph_input_embd>(hparams.n_embd_inp_enc());
    inp_target->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp_enc(), n_tokens);
    ggml_set_input(inp_target->embd);

    ggml_tensor * cur = inp_target->embd;
    cb(cur, "dspark_target_hidden", -1);
    res->add_input(std::move(inp_target));
    return cur;
}

template <>
llama_model_deepseek4_dspark_draft::graph<true>::graph(
        const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = build_inp_embd_enc();

    // Official DSpark: main_proj [3*4096,4096] followed by main_norm.
    cur = build_lora_mm(model.fc, cur);
    cb(cur, "dspark_main_proj", -1);
    cur = build_norm(cur, model.output_norm_enc, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "dspark_main_norm", -1);

    ggml_set_output(cur);
    res->t_h_nextn = cur;
    ggml_build_forward_expand(gf, cur);
}

static size_t dspark_row_offset(const ggml_tensor * t, int64_t row) {
    return ggml_row_size(t->type, row);
}

static ggml_tensor * dspark_view_1d(ggml_context * ctx, ggml_tensor * t, int64_t n, int64_t row) {
    return ggml_view_1d(ctx, t, n, dspark_row_offset(t, row));
}

static ggml_tensor * dspark_view_2d(ggml_context * ctx, ggml_tensor * t,
        int64_t ne0, int64_t ne1, int64_t row) {
    return ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], dspark_row_offset(t, row));
}

// Keep the official DSpark hyperconnection contract in graph form.  In
// particular, do not replace this with a plain residual add: the [pre, post,
// combine] tensors are learned parameters and are part of the drafter recipe.
static ggml_tensor * dspark_hc_pre(llm_graph_context & g, ggml_tensor * x,
        ggml_tensor * fn, ggml_tensor * scale, ggml_tensor * base,
        ggml_tensor ** post, ggml_tensor ** combine, int il) {
    const int64_t hc = g.hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[2];
    const int64_t hc_dim = hc * g.n_embd;
    const int64_t mix_dim = (2 + hc) * hc;

    ggml_tensor * flat = ggml_reshape_2d(g.ctx0, x, hc_dim, nt);
    ggml_tensor * mixes = ggml_mul_mat(g.ctx0, fn, ggml_rms_norm(g.ctx0, flat, g.norm_rms_eps));
    g.cb(mixes, "dspark_hc_mixes", il);

    ggml_tensor * pre = dspark_view_2d(g.ctx0, mixes, hc, nt, 0);
    pre = ggml_add(g.ctx0, ggml_mul(g.ctx0, pre, dspark_view_1d(g.ctx0, scale, 1, 0)),
                   dspark_view_1d(g.ctx0, base, hc, 0));
    pre = ggml_scale_bias(g.ctx0, ggml_sigmoid(g.ctx0, pre), 1.0f, g.hparams.dsv4_hc_eps);

    *post = dspark_view_2d(g.ctx0, mixes, hc, nt, hc);
    *post = ggml_add(g.ctx0, ggml_mul(g.ctx0, *post, dspark_view_1d(g.ctx0, scale, 1, 1)),
                     dspark_view_1d(g.ctx0, base, hc, hc));
    *post = ggml_scale(g.ctx0, ggml_sigmoid(g.ctx0, *post), 2.0f);

    ggml_tensor * comb = dspark_view_2d(g.ctx0, mixes, hc * hc, nt, 2 * hc);
    comb = ggml_add(g.ctx0, ggml_mul(g.ctx0, comb, dspark_view_1d(g.ctx0, scale, 1, 2)),
                    dspark_view_1d(g.ctx0, base, hc * hc, 2 * hc));
    comb = ggml_reshape_3d(g.ctx0, comb, hc, hc, nt);
    comb = ggml_soft_max(g.ctx0, comb);
    ggml_tensor * eps = ggml_new_tensor_1d(g.ctx0, GGML_TYPE_F32, 1);
    eps = ggml_fill(g.ctx0, eps, g.hparams.dsv4_hc_eps);
    comb = ggml_add(g.ctx0, comb, eps);
    auto norm_cols = [&]() {
        ggml_tensor * trans = ggml_cont(g.ctx0, ggml_permute(g.ctx0, comb, 1, 0, 2, 3));
        ggml_tensor * sum = ggml_add(g.ctx0, ggml_sum_rows(g.ctx0, trans), eps);
        sum = ggml_permute(g.ctx0, sum, 1, 0, 2, 3);
        comb = ggml_div(g.ctx0, comb, sum);
    };
    auto norm_rows = [&]() {
        ggml_tensor * sum = ggml_add(g.ctx0, ggml_sum_rows(g.ctx0, comb), eps);
        comb = ggml_div(g.ctx0, comb, sum);
    };
    norm_cols();
    for (uint32_t i = 1; i < g.hparams.dsv4_hc_sinkhorn_iters; ++i) {
        norm_rows();
        norm_cols();
    }
    *combine = comb;

    GGML_ASSERT(fn->ne[1] == mix_dim);
    ggml_tensor * out = nullptr;
    for (int64_t ih = 0; ih < hc; ++ih) {
        ggml_tensor * xh = ggml_view_2d(g.ctx0, x, g.n_embd, nt, x->nb[2], ih * x->nb[1]);
        ggml_tensor * ph = ggml_view_2d(g.ctx0, pre, 1, nt, pre->nb[1], ih * pre->nb[0]);
        ggml_tensor * cur = ggml_mul(g.ctx0, xh, ph);
        // HC pre contracts the stream axis before the transformer block.  The
        // post/combine path expands the block output back to all streams.
        out = out ? ggml_add(g.ctx0, out, cur) : cur;
    }
    return out;
}

static ggml_tensor * dspark_attn(llm_graph_context & g, const llama_layer & layer,
        llm_graph_input_attn_kv * inp_attn, ggml_tensor * x, ggml_tensor * pos, int il) {
    const int64_t head = g.hparams.n_embd_head_k();
    ggml_tensor * qa = g.build_lora_mm(layer.wq_a, x);
    qa = g.build_norm(qa, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    ggml_tensor * q = g.build_lora_mm(layer.wq_b, qa);
    q = ggml_reshape_3d(g.ctx0, q, head, g.n_head, g.n_tokens);

    ggml_tensor * kv = g.build_lora_mm(layer.wkv, x);
    kv = g.build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
    ggml_tensor * k = ggml_reshape_3d(g.ctx0, kv, head, g.n_head_kv, g.n_tokens);
    ggml_tensor * v = ggml_reshape_3d(g.ctx0, kv, head, g.n_head_kv, g.n_tokens);

    q = ggml_rope_ext(g.ctx0, q, pos, nullptr, g.n_rot, g.rope_type, g.n_ctx_orig,
            g.freq_base, g.freq_scale, g.ext_factor, g.attn_factor, g.beta_fast, g.beta_slow);
    k = ggml_rope_ext(g.ctx0, k, pos, nullptr, g.n_rot, g.rope_type, g.n_ctx_orig,
            g.freq_base, g.freq_scale, g.ext_factor, g.attn_factor, g.beta_fast, g.beta_slow);
    ggml_tensor * out = g.build_attn(inp_attn, nullptr, nullptr, nullptr, q, k, v,
            nullptr, layer.attn_sinks, nullptr, 1.0f / sqrtf(float(head)), il);

    // DSpark uses grouped low-rank output projection, matching DeepSeek-V4's
    // output_a/output_b contract (8 groups, rank 1024).
    const int64_t groups = g.hparams.dsv4_o_group_count;
    const int64_t rank = g.hparams.dsv4_o_lora_rank;
    const int64_t group_dim = head * g.n_head / groups;
    out = ggml_reshape_3d(g.ctx0, out, group_dim, groups, g.n_tokens);
    out = ggml_permute(g.ctx0, out, 0, 2, 1, 3);
    ggml_tensor * oa = ggml_mul_mat(g.ctx0,
            ggml_reshape_3d(g.ctx0, layer.wo_a, layer.wo_a->ne[0], rank, groups), out);
    oa = ggml_permute(g.ctx0, oa, 0, 2, 1, 3);
    oa = ggml_cont_2d(g.ctx0, oa, rank * groups, g.n_tokens);
    return g.build_lora_mm(layer.wo_b, oa);
}

// Apply the official DSpark Markov and confidence heads to the decoder
// logits.  The heads are block-oriented: each sequence contributes one
// equal-size block, and the Markov chain conditions each position on the
// anchor (or the previous greedy position) in that block.
static void build_dspark_heads(llm_graph_context & g, const llama_model & model,
        ggml_tensor * tokens, ggml_tensor * hidden) {
    GGML_ASSERT(model.dspark_markov_w1 && model.dspark_markov_w2 && model.dspark_conf_proj);

    ggml_context * ctx0 = g.ctx0;
    ggml_tensor * base = g.res->t_logits;
    const int64_t n_vocab = base->ne[0];
    const int64_t n_tok   = base->ne[1];

    auto it = model.gguf_kv.find("dspark.block_size");
    GGML_ASSERT(it != model.gguf_kv.end() && "DSpark requires dspark.block_size metadata");
    const int64_t block_size = std::stoll(it->second);
    const int64_t n_blocks = g.ubatch.n_seqs_unq;
    GGML_ASSERT(block_size > 0 && n_blocks > 0 && n_tok % n_blocks == 0 &&
            "DSpark Markov head requires equal-size blocks");

    const int64_t block_drafts = n_tok / n_blocks;
    // Reserve graphs may use the full context/batch size rather than the
    // trained DSpark block.  Keep base logits for that sizing pass; the
    // Markov/confidence chain is rebuilt when the runtime block is <= block_size.
    if (block_drafts > block_size) {
        return;
    }

    const size_t token_stride = (size_t) block_drafts * tokens->nb[0];
    const size_t base_stride  = (size_t) block_drafts * base->nb[1];
    ggml_tensor * prev = ggml_cont_1d(ctx0,
            ggml_view_2d(ctx0, tokens, 1, n_blocks, token_stride, 0), n_blocks);

    ggml_tensor * logits = nullptr;
    ggml_tensor * confs = nullptr;
    const int64_t rank = model.dspark_markov_w1->ne[0];
    for (int64_t i = 0; i < block_drafts; ++i) {
        ggml_tensor * w1_prev = ggml_get_rows(ctx0, model.dspark_markov_w1, prev);
        ggml_tensor * bias = ggml_mul_mat(ctx0, model.dspark_markov_w2, w1_prev);
        ggml_tensor * base_i = ggml_view_2d(ctx0, base, n_vocab, n_blocks,
                base_stride, i * base->nb[1]);
        ggml_tensor * col = ggml_add(ctx0, base_i, bias);
        logits = logits ? ggml_concat(ctx0, logits, col, 1) : col;

        ggml_tensor * hidden_i = ggml_view_2d(ctx0, hidden, hidden->ne[0], n_blocks,
                (size_t) block_drafts * hidden->nb[1], i * hidden->nb[1]);
        ggml_tensor * feat = ggml_concat(ctx0, ggml_cont(ctx0, hidden_i), w1_prev, 0);
        GGML_ASSERT(feat->ne[0] == g.n_embd + rank);
        ggml_tensor * conf = ggml_sigmoid(ctx0, ggml_mul_mat(ctx0, model.dspark_conf_proj, feat));
        confs = confs ? ggml_concat(ctx0, confs, conf, 1) : conf;

        if (i + 1 < block_drafts) {
            prev = ggml_argmax(ctx0, col);
        }
    }

    logits = ggml_reshape_3d(ctx0, logits, n_vocab, n_blocks, block_drafts);
    logits = ggml_reshape_2d(ctx0,
            ggml_cont(ctx0, ggml_permute(ctx0, logits, 0, 2, 1, 3)), n_vocab, n_tok);
    g.res->t_logits = logits;
    ggml_build_forward_expand(g.gf, logits);

    confs = ggml_reshape_3d(ctx0, confs, 1, n_blocks, block_drafts);
    confs = ggml_reshape_2d(ctx0,
            ggml_cont(ctx0, ggml_permute(ctx0, confs, 0, 2, 1, 3)), 1, n_tok);
    // Reuse the nextn API's fixed hidden-width storage for per-position
    // confidence, as the generic DFlash implementation does.
    g.res->t_h_nextn = ggml_repeat(ctx0, confs, hidden);
    ggml_build_forward_expand(g.gf, g.res->t_h_nextn);
}

template <>
llama_model_deepseek4_dspark_draft::graph<false>::graph(
        const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    // The official drafter has no vocabulary.  build_inp_embd needs a real
    // token matrix, so resolve it from the target context rather than creating
    // a second (incorrect) vocabulary in the DSpark model.
    const llama_model * target = &model;
    if (target->tok_embd == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr && "DSpark decoder requires ctx_other");
        target = llama_get_model(cparams.ctx_other);
        GGML_ASSERT(target != nullptr && target->tok_embd != nullptr &&
                "DSpark decoder requires target token embeddings");
    }
    GGML_ASSERT(cparams.ctx_other != nullptr || model.output != nullptr);
    const ggml_tensor * target_output = model.output;
    if (target_output == nullptr) {
        target_output = target->output;
        GGML_ASSERT(target_output != nullptr && "DSpark decoder requires target output projection");
    }
    // A tensor-split target exposes a mirrored output projection through the
    // Meta buffer. The draft scheduler uses concrete device backends, so use
    // one full mirrored copy rather than passing the Meta wrapper into it.
    if (target_output->buffer != nullptr && ggml_backend_buffer_is_meta(target_output->buffer)) {
        target_output = ggml_backend_meta_get_simple_tensor(target_output, 0);
    }

    ggml_tensor * inp = build_inp_embd(target->tok_embd);
    ggml_tensor * pos = build_inp_pos();
    llm_graph_input_attn_kv * inp_attn = build_attn_inp_kv();
    ggml_build_forward_expand(gf, inp_attn->self_kq_mask);

    const int64_t hc = hparams.dsv4_hc_mult;
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);

    const char * pin_env = getenv("GGML_DSPARK_PIN_OUTPUTS");
    const bool pin_layer_outputs = pin_env != nullptr && pin_env[0] != '0';

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * combine = nullptr;
        ggml_tensor * cur = dspark_hc_pre(*this, inpL, layer.hc_attn_fn,
                layer.hc_attn_scale, layer.hc_attn_base, &post, &combine, il);
        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cur = dspark_attn(*this, layer, inp_attn, cur, pos, il);
        inpL = [&]() {
            ggml_tensor * out = nullptr;
            for (int64_t dst = 0; dst < hc; ++dst) {
                ggml_tensor * p = ggml_view_2d(ctx0, post, 1, n_tokens, post->nb[1], dst * post->nb[0]);
                ggml_tensor * z = ggml_mul(ctx0, cur, p);
                for (int64_t src = 0; src < hc; ++src) {
                    ggml_tensor * r = ggml_view_2d(ctx0, residual, n_embd, n_tokens, residual->nb[2], src * residual->nb[1]);
                    ggml_tensor * c = ggml_view_2d(ctx0, combine, 1, n_tokens, combine->nb[2], dst * combine->nb[0] + src * combine->nb[1]);
                    z = ggml_add(ctx0, z, ggml_mul(ctx0, r, c));
                }
                z = ggml_reshape_3d(ctx0, z, n_embd, 1, n_tokens);
                out = out ? ggml_concat(ctx0, out, z, 1) : z;
            }
            return out;
        }();

        residual = inpL;
        cur = dspark_hc_pre(*this, inpL, layer.hc_ffn_fn,
                layer.hc_ffn_scale, layer.hc_ffn_base, &post, &combine, il);

        // Optionally keep the residual and hyperconnection tensors on the graph
        // explicitly.  Pinning them as roots was a workaround for a scheduler
        // hang on small GPU batches, but it forces extra graph splits and
        // cross-device syncs for a layer-split draft.  Re-enable with
        // GGML_DSPARK_PIN_OUTPUTS=1 if the hang reappears.
        if (pin_layer_outputs) {
            ggml_build_forward_expand(gf, residual);
            ggml_build_forward_expand(gf, post);
            ggml_build_forward_expand(gf, combine);
        }

        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        ggml_tensor * moe = build_moe_ffn(cur, layer.ffn_gate_inp,
                layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_down_exps,
                layer.ffn_exp_probs_b, n_expert, n_expert_used, LLM_FFN_SILU,
                hparams.expert_weights_norm, hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func, il);
        ggml_tensor * shared = build_ffn(cur, layer.ffn_up_shexp, nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr, layer.ffn_down_shexp,
                nullptr, nullptr, nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cur = ggml_add(ctx0, moe, shared);
        inpL = [&]() {
            ggml_tensor * out = nullptr;
            for (int64_t dst = 0; dst < hc; ++dst) {
                ggml_tensor * p = ggml_view_2d(ctx0, post, 1, n_tokens, post->nb[1], dst * post->nb[0]);
                ggml_tensor * z = ggml_mul(ctx0, cur, p);
                for (int64_t src = 0; src < hc; ++src) {
                    ggml_tensor * r = ggml_view_2d(ctx0, residual, n_embd, n_tokens, residual->nb[2], src * residual->nb[1]);
                    ggml_tensor * c = ggml_view_2d(ctx0, combine, 1, n_tokens, combine->nb[2], dst * combine->nb[0] + src * combine->nb[1]);
                    z = ggml_add(ctx0, z, ggml_mul(ctx0, r, c));
                }
                z = ggml_reshape_3d(ctx0, z, n_embd, 1, n_tokens);
                out = out ? ggml_concat(ctx0, out, z, 1) : z;
            }
            return out;
        }();
        inpL = build_cvec(inpL, il);
        cb(inpL, "l_last", il);
    }

    ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd * hc, n_tokens);
    ggml_tensor * mixes = ggml_mul_mat(ctx0, model.hc_head_fn,
            ggml_rms_norm(ctx0, flat, norm_rms_eps));
    ggml_tensor * head_pre = ggml_add(ctx0,
            ggml_mul(ctx0, mixes, model.hc_head_scale), model.hc_head_base);
    head_pre = ggml_scale_bias(ctx0, ggml_sigmoid(ctx0, head_pre), 1.0f, hparams.dsv4_hc_eps);
    ggml_tensor * head = [&]() {
        ggml_tensor * out = nullptr;
        for (int64_t ih = 0; ih < hc; ++ih) {
            ggml_tensor * xh = ggml_view_2d(ctx0, flat, n_embd, n_tokens, flat->nb[1], ih * n_embd);
            ggml_tensor * wh = ggml_view_2d(ctx0, head_pre, 1, n_tokens, head_pre->nb[1], ih);
            ggml_tensor * z = ggml_mul(ctx0, xh, wh);
            out = out ? ggml_add(ctx0, out, z) : z;
        }
        return out;
    }();
    head = build_norm(head, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    res->t_embd = head;
    res->t_logits = build_lora_mm(const_cast<ggml_tensor *>(target_output), head);
    ggml_build_forward_expand(gf, res->t_logits);

    // The official artifact's Markov/confidence tensors are part of the
    // decoder contract, rather than optional metadata.  They also require
    // the complete trained block submitted by common/speculative.cpp.
    // KV-cache injection supplies hidden embeddings without token ids; do not
    // feed its uninitialized token input into the Markov get-rows chain.
    if (ubatch.token != nullptr) {
        build_dspark_heads(*this, model, res->get_inp_tokens(), head);
    }
}
