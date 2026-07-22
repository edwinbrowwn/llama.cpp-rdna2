#include "server-sequence-fork-policy.h"
#include "llama-ext.h"
#include "ggml.h"

#include <cstdio>
#include <vector>

#define CHECK(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        return 1; \
    } \
} while (0)

int main() {
    using namespace server_sequence_fork_policy;

    // The scoped backend contract is encoded in each FA operation, not global
    // process state. Validate the op-parameter round trip without a backend.
    ggml_init_params iparams = {
        /*.mem_size   =*/ 1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(iparams);
    CHECK(ctx != nullptr);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 1, 1, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 64, 256, 1, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 64, 256, 1, 1);
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 256, 1, 1, 1);
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f, 0.0f, 0.0f);
    CHECK(fa != nullptr);
    CHECK(!ggml_flash_attn_ext_get_force_vec(fa));
    ggml_flash_attn_ext_set_force_vec(fa, true);
    CHECK(ggml_flash_attn_ext_get_force_vec(fa));
    ggml_flash_attn_ext_set_force_vec(fa, false);
    CHECK(!ggml_flash_attn_ext_get_force_vec(fa));
    ggml_free(ctx);

    auto make_model = [](uint32_t dk, uint32_t dv) {
        llama_quant_model_desc desc = {};
        desc.architecture = "llama";
        desc.n_embd = 4096;
        desc.n_ff = 11008;
        desc.n_layer = 2;
        desc.n_head = 16;
        desc.n_head_kv = 2;
        desc.n_embd_head_k = dk;
        desc.n_embd_head_v = dv;
        return llama_quant_model_from_metadata(&desc);
    };

    llama_model * model = make_model(256, 256);
    CHECK(model != nullptr);
    CHECK(llama_model_supports_flash_attn_force_vec(model));
    llama_model_free(model);

    model = make_model(96, 96);
    CHECK(model != nullptr);
    CHECK(!llama_model_supports_flash_attn_force_vec(model));
    llama_model_free(model);

    model = make_model(256, 128);
    CHECK(model != nullptr);
    CHECK(!llama_model_supports_flash_attn_force_vec(model));
    llama_model_free(model);

    CHECK(!llama_model_supports_flash_attn_force_vec(nullptr));

    // Conservative 1/12 threshold, including the exact boundary.
    CHECK(should_use_vector(true, true, 84, 77));
    CHECK(!should_use_vector(true, true, 84, 76));
    CHECK(should_use_vector(true, true, 62781, 59126));
    CHECK(!should_use_vector(true, true, 26917, 11538));

    // Restored tile FA is never selected as an AMD fallback when vector is unavailable.
    CHECK(!should_use_vector(true, false, 80, 79));
    CHECK(!should_use_vector(false, true, 80, 79));
    CHECK(!should_use_vector(true, true, 80, 0));
    CHECK(!should_use_vector(true, true, 0, 0));
    CHECK(!should_use_vector(true, true, 80, 81));

    // An exact boundary remains eligible; the server may add one mandatory
    // prompt-logit reevaluation token after selecting the policy.
    CHECK(should_use_vector(true, true, 80, 80));

    restored_suffix_state state;
    CHECK(!state.active());
    CHECK(!state.force_vector());
    CHECK(state.start(10));
    CHECK(state.active());
    CHECK(state.force_vector());
    CHECK(state.remaining() == 10);
    CHECK(state.consume(4));
    CHECK(state.remaining() == 6);
    CHECK(!state.consume(7));
    CHECK(state.remaining() == 6);
    CHECK(state.consume(6));
    CHECK(!state.active());
    CHECK(state.remaining() == 0);
    CHECK(state.force_vector());

    // Task release clears the prompt counter but the restored sequence remains
    // tainted until its memory is atomically rebuilt.
    CHECK(state.start(3));
    state.clear_prompt();
    CHECK(!state.active());
    CHECK(state.force_vector());

    // Prompt clear/full reprocess clears the underlying restored provenance.
    state.clear_all();
    CHECK(!state.active());
    CHECK(!state.force_vector());

    CHECK(!state.start(0));
    CHECK(!state.active());
    CHECK(!state.force_vector());
    CHECK(!state.start((size_t) UINT32_MAX + 1));
    CHECK(!state.active());
    CHECK(!state.force_vector());

    CHECK(state.start(5));
    restored_suffix_state copied = state;
    CHECK(copied.active());
    CHECK(copied.force_vector());
    CHECK(copied.remaining() == 5);

    const std::vector<bool> mixed = {true, true, false, false};
    CHECK(homogeneous_prefix(mixed.size(), [&](size_t i) { return mixed[i]; }) == 2);
    CHECK(homogeneous_prefix(mixed.size(), [&](size_t i) { return !mixed[i]; }) == 2);

    const std::vector<bool> all_vector = {true, true, true};
    CHECK(homogeneous_prefix(all_vector.size(), [&](size_t i) { return all_vector[i]; }) == 3);

    const std::vector<bool> split_immediately = {false, true, true};
    CHECK(homogeneous_prefix(split_immediately.size(), [&](size_t i) { return split_immediately[i]; }) == 1);
    CHECK(homogeneous_prefix(0, [&](size_t) { return true; }) == 0);

    std::puts("test-sequence-fork-policy: PASS");
    return 0;
}