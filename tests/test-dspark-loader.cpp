#include "llama.h"
#include "../src/llama-ext.h"

#include <cstdio>

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s DSPARK_DRAFT.gguf\n", argv[0]);
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

    // The official artifact captures exactly these target layers.  This also
    // proves the loader/data boundary is not silently accepting a different
    // feature ordering.
    const int32_t * target_layers = llama_model_target_layer_ids(model);
    if (llama_model_target_layer_ids_n(model) != 3 || target_layers == nullptr ||
            target_layers[0] != 40 || target_layers[1] != 41 || target_layers[2] != 42) {
        std::fprintf(stderr, "DeepSeek-V4 DSpark target-layer boundary mismatch\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // Decoder construction is intentionally an explicit boundary: target
    // hidden-state handoff and DeepSeek-V4 KV-cache wiring are not complete.
    auto context_params = llama_context_default_params();
    context_params.n_ctx = 1;
    context_params.n_batch = 1;
    llama_context * context = llama_init_from_model(model, context_params);
    const bool reached_graph_boundary = context == nullptr;
    if (context != nullptr) {
        llama_free(context);
    }
    llama_model_free(model);
    llama_backend_free();

    if (!reached_graph_boundary) {
        std::fprintf(stderr, "DeepSeek-V4 DSpark graph unexpectedly initialized\n");
        return 1;
    }
    return 0;
}