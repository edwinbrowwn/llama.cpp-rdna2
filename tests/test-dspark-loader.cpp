#include "llama.h"
#include "../src/llama-ext.h"

#include <cstdio>

int main(int argc, char ** argv) {
    if (argc != 2 && argc != 3) {
        std::fprintf(stderr, "usage: %s DSPARK_DRAFT.gguf [TARGET.gguf]\n", argv[0]);
        return 2;
    }

    llama_backend_init();
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(argv[1], model_params);
    if (model == nullptr) {
        std::fprintf(stderr, "DeepSeek-V4 DSpark loader failed\n");
        llama_backend_free();
        return 1;
    }

    const int32_t * target_layers = llama_model_target_layer_ids(model);
    if (llama_model_target_layer_ids_n(model) != 3 || target_layers == nullptr ||
            target_layers[0] != 40 || target_layers[1] != 41 || target_layers[2] != 42) {
        std::fprintf(stderr, "DeepSeek-V4 DSpark target-layer boundary mismatch\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    llama_model * target_model = nullptr;
    llama_context * target_context = nullptr;
    if (argc == 3) {
        target_model = llama_model_load_from_file(argv[2], model_params);
        if (target_model == nullptr) {
            std::fprintf(stderr, "target model loader failed\n");
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        auto target_params = llama_context_default_params();
        target_params.n_ctx = 16;
        target_params.n_batch = 16;
        target_context = llama_init_from_model(target_model, target_params);
        if (target_context == nullptr) {
            std::fprintf(stderr, "target context initialization failed\n");
            llama_model_free(target_model);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
    }

    auto context_params = llama_context_default_params();
    context_params.n_ctx = 16;
    context_params.n_batch = 16;
    context_params.ctx_other = target_context;
    llama_context * context = llama_init_from_model(model, context_params);

    // Without ctx_other the official artifact must reject decoder construction,
    // because it intentionally has no vocabulary/output.  With a target model,
    // successful initialization proves the decoder graph resolved both shared
    // tensors and its official three-layer recipe.
    const bool expected = argc == 3;
    const bool reached_graph = context != nullptr;
    if (reached_graph == expected && expected) {
        constexpr int32_t n_block = 5;
        llama_batch batch = llama_batch_init(n_block, 0, 1);
        const llama_vocab * vocab = llama_model_get_vocab(target_model);
        const llama_token token = llama_vocab_bos(vocab);
        if (token == LLAMA_TOKEN_NULL) {
            std::fprintf(stderr, "target vocabulary has no BOS token\n");
            llama_batch_free(batch);
            if (context) llama_free(context);
            if (target_context) llama_free(target_context);
            if (target_model) llama_model_free(target_model);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        for (int32_t i = 0; i < n_block; ++i) {
            batch.token[i] = token;
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = i == n_block - 1;
        }
        batch.n_tokens = n_block;
        const int rc = llama_decode(context, batch);
        if (rc == 0) {
            llama_synchronize(context);
            if (llama_get_logits_ith(context, n_block - 1) == nullptr) {
                std::fprintf(stderr, "DSpark decoder produced no logits\n");
                llama_batch_free(batch);
                if (context) llama_free(context);
                if (target_context) llama_free(target_context);
                if (target_model) llama_model_free(target_model);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }
        }
        llama_batch_free(batch);
        if (rc != 0) {
            std::fprintf(stderr, "DSpark decoder graph execution failed rc=%d\n", rc);
            if (context) llama_free(context);
            if (target_context) llama_free(target_context);
            if (target_model) llama_model_free(target_model);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
    }
    if (reached_graph != expected) {
        std::fprintf(stderr, "unexpected DSpark graph construction result (got=%d expected=%d)\n",
                (int) reached_graph, (int) expected);
        if (context) llama_free(context);
        if (target_context) llama_free(target_context);
        if (target_model) llama_model_free(target_model);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    if (context) llama_free(context);
    if (target_context) llama_free(target_context);
    if (target_model) llama_model_free(target_model);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
