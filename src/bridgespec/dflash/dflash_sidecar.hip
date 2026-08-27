// DFlash2 HIP drafter sidecar DLL — loaded by llama.cpp's impl_draft_dflash.
// SPDX-License-Identifier: MIT
// API: spec_dflash_init(dir) / spec_dflash_chunk(poss, feats, n) /
//      spec_dflash_draft(id_last, n_past, out_ids[7]).
#include "dflash_kernels.hip"
#include "../artifact_manifest.h"

#define DFC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "[dflash-sidecar] %s @%d\n", hipGetErrorString(e_), __LINE__); return -1; } } while (0)

using Tens = bridgespec_artifact::TensorDesc;

static void deq_q4k_host(const uint8_t * src, float * dst, size_t n) {
    const block_q4_K * bs = (const block_q4_K *) src;
    for (size_t b = 0; b < n / 256; ++b) {
        const block_q4_K * bk = &bs[b];
        const float d = df_f16(bk->d), dmin = df_f16(bk->dmin);
        float * out = dst + b * 256;
        for (int g64 = 0; g64 < 4; ++g64) {
            float sc0, mn0, sc1, mn1;
            q4k_scale_min(g64 * 2 + 0, bk->scales, &sc0, &mn0);
            q4k_scale_min(g64 * 2 + 1, bk->scales, &sc1, &mn1);
            const uint8_t * qs = bk->qs + g64 * 32;
            for (int l = 0; l < 32; ++l) {
                out[g64 * 64 + l]      = d * sc0 * (qs[l] & 0xF) - dmin * mn0;
                out[g64 * 64 + 32 + l] = d * sc1 * (qs[l] >> 4)  - dmin * mn1;
            }
        }
    }
}

#define DISPATCH_Q4K(N, W, X, Y, NIN, NOUT, ST) do { \
    switch (N) { \
    case 1: gemv8t_q4_K<1><<<((NOUT)+7)/8, 256, 0, ST>>>(W, X, Y, NIN, NOUT); break; \
    case 2: gemv8t_q4_K<2><<<((NOUT)+7)/8, 256, 0, ST>>>(W, X, Y, NIN, NOUT); break; \
    case 3: gemv8t_q4_K<3><<<((NOUT)+7)/8, 256, 0, ST>>>(W, X, Y, NIN, NOUT); break; \
    case 4: gemv8t_q4_K<4><<<((NOUT)+7)/8, 256, 0, ST>>>(W, X, Y, NIN, NOUT); break; \
    case 5: gemv8t_q4_K<5><<<((NOUT)+7)/8, 256, 0, ST>>>(W, X, Y, NIN, NOUT); break; \
    case 6: gemv8t_q4_K<6><<<((NOUT)+7)/8, 256, 0, ST>>>(W, X, Y, NIN, NOUT); break; \
    case 7: gemv8t_q4_K<7><<<((NOUT)+7)/8, 256, 0, ST>>>(W, X, Y, NIN, NOUT); break; \
    default: gemv8t_q4_K<8><<<((NOUT)+7)/8, 256, 0, ST>>>(W, X, Y, NIN, NOUT); break; } } while (0)
#define DISPATCH_Q6K(N, W, X, Y, NIN, NOUT, ST) do { \
    switch (N) { \
    case 1: gemv8t_q6_K<1><<<((NOUT)+7)/8, 256, 0, ST>>>(W, X, Y, NIN, NOUT); break; \
    case 2: gemv8t_q6_K<2><<<((NOUT)+7)/8, 256, 0, ST>>>(W, X, Y, NIN, NOUT); break; \
    case 3: gemv8t_q6_K<3><<<((NOUT)+7)/8, 256, 0, ST>>>(W, X, Y, NIN, NOUT); break; \
    case 4: gemv8t_q6_K<4><<<((NOUT)+7)/8, 256, 0, ST>>>(W, X, Y, NIN, NOUT); break; \
    case 5: gemv8t_q6_K<5><<<((NOUT)+7)/8, 256, 0, ST>>>(W, X, Y, NIN, NOUT); break; \
    case 6: gemv8t_q6_K<6><<<((NOUT)+7)/8, 256, 0, ST>>>(W, X, Y, NIN, NOUT); break; \
    case 7: gemv8t_q6_K<7><<<((NOUT)+7)/8, 256, 0, ST>>>(W, X, Y, NIN, NOUT); break; \
    default: gemv8t_q6_K<8><<<((NOUT)+7)/8, 256, 0, ST>>>(W, X, Y, NIN, NOUT); break; } } while (0)

namespace {
struct DLayer {
    const block_q4_K *conv_a_proj, *conv_f_proj, *wq, *wk, *wo, *up, *gate;
    const void * wv; bool wv_q6;
    const void * down; bool down_q6;
    const float *attn_norm, *qn, *kn, *ffn_norm, *conv_a_base, *conv_f_base;
};
struct DState {
    bool ok = false;
    std::vector<Tens> G;
    hipStream_t st;
    DLayer L[DF_LAYERS];
    const block_q4_K *W_fc, *W_selhid;
    const block_q4_0d *W_embd;
    const block_q6_K *W_head;
    const float *w_enc_norm, *w_out_norm;
    std::vector<float> sel_prev, sel_next;
    std::vector<int32_t> remap;
    int n_head = 40960; // rows in the loaded target head (sliced or full)
    float *feat_d, *henc, *x, *xn, *dyn_a, *dyn_f, *cg, *q, *k, *v, *att, *wo_out,
          *fu, *fg, *ftmp, *resid, *tE, *logits, *selhid, *tk_pmax, *tk_vals;
    int *pos_d, *tok_d, *tk_pidx, *tk_ids;
    __half *Kc[DF_LAYERS], *Vc[DF_LAYERS];
    int max_pos = 16384;
    int tkparts = 0;
};
DState S;
Tens * DT(const std::string & n) {
    return bridgespec_artifact::find_tensor(S.G, n.c_str());
}
} // namespace

#ifdef _WIN32
#define DF_API extern "C" __declspec(dllexport)
#else
#define DF_API extern "C"
#endif

static constexpr int32_t DF_RELEASE_ABI = 2;

DF_API int spec_dflash_release_abi() {
    return DF_RELEASE_ABI;
}

DF_API int spec_dflash_check(int32_t n_embd_enc, int32_t block_size) {
    return (n_embd_enc == 25600 && block_size == DF_BLOCK) ? 0 : -1;
}

static bool expect_tensor(const std::vector<Tens> & tensors, const std::string & name,
                          const char * dtype, std::initializer_list<int64_t> shape,
                          uint64_t nbytes, std::string & error) {
    const Tens * tensor = bridgespec_artifact::find_tensor(tensors, name.c_str());
    if (tensor == nullptr) {
        error = "missing tensor: " + name;
        return false;
    }
    if (!bridgespec_artifact::tensor_matches(*tensor, dtype, shape, nbytes)) {
        error = "tensor schema mismatch: " + name;
        return false;
    }
    return true;
}

static bool validate_dflash_schema(const std::vector<Tens> & tensors, std::string & error) {
    if (tensors.size() != 81) {
        error = "DFlash manifest must contain exactly 81 tensors";
        return false;
    }
    if (!expect_tensor(tensors, "enc.output_norm.weight",       "0",  {5120},          20480, error) ||
        !expect_tensor(tensors, "fc.weight",                    "12", {25600, 5120}, 73728000, error) ||
        !expect_tensor(tensors, "output_norm.weight",           "0",  {5120},          20480, error) ||
        !expect_tensor(tensors, "selector_hidden.weight",       "12", {5120, 256},     737280, error) ||
        !expect_tensor(tensors, "selector_predecessor.weight",  "12", {256, 248320}, 35758080, error) ||
        !expect_tensor(tensors, "selector_successor.weight",    "12", {256, 248320}, 35758080, error)) {
        return false;
    }
    for (int layer = 0; layer < DF_LAYERS; ++layer) {
        char prefix[32];
        std::snprintf(prefix, sizeof(prefix), "blk.%d.", layer);
        const auto name = [&](const char * suffix) { return std::string(prefix) + suffix; };
        const bool q6 = layer == 2 || layer == 4;
        if (!expect_tensor(tensors, name("attn_conv_base"),        "0",  {5120, 2, 2}, 81920, error) ||
            !expect_tensor(tensors, name("attn_conv_proj.weight"), "12", {5120, 1280}, 3686400, error) ||
            !expect_tensor(tensors, name("attn_k.weight"),         "12", {5120, 1024}, 2949120, error) ||
            !expect_tensor(tensors, name("attn_k_norm.weight"),    "0",  {128}, 512, error) ||
            !expect_tensor(tensors, name("attn_norm.weight"),      "0",  {5120}, 20480, error) ||
            !expect_tensor(tensors, name("attn_output.weight"),    "12", {4096, 5120}, 11796480, error) ||
            !expect_tensor(tensors, name("attn_q.weight"),         "12", {5120, 4096}, 11796480, error) ||
            !expect_tensor(tensors, name("attn_q_norm.weight"),    "0",  {128}, 512, error) ||
            !expect_tensor(tensors, name("attn_v.weight"), q6 ? "14" : "12", {5120, 1024}, q6 ? 4300800 : 2949120, error) ||
            !expect_tensor(tensors, name("ffn_conv_base"),         "0",  {5120, 2, 2}, 81920, error) ||
            !expect_tensor(tensors, name("ffn_conv_proj.weight"),  "12", {5120, 1280}, 3686400, error) ||
            !expect_tensor(tensors, name("ffn_down.weight"), q6 ? "14" : "12", {17408, 5120}, q6 ? 73113600 : 50135040, error) ||
            !expect_tensor(tensors, name("ffn_gate.weight"),       "12", {5120, 17408}, 50135040, error) ||
            !expect_tensor(tensors, name("ffn_norm.weight"),       "0",  {5120}, 20480, error) ||
            !expect_tensor(tensors, name("ffn_up.weight"),         "12", {5120, 17408}, 50135040, error)) {
            return false;
        }
    }
    return true;
}

static bool validate_target_schema(const std::vector<Tens> & tensors, std::string & error) {
    struct Exp { const char * name, * dtype; int64_t ne0, ne1; uint64_t nbytes; };
    static const Exp EXPECTED[] = {
        {"token_embd.weight",                         "2",  5120, 248320, 715161600},
        {"blk.64.attn_k.weight",                     "2",  5120,   1024,   2949120},
        {"blk.64.attn_k_norm.weight",                "0",   256,      0,      1024},
        {"blk.64.attn_norm.weight",                  "0",  5120,      0,     20480},
        {"blk.64.attn_output.weight",                "2",  6144,   5120,  17694720},
        {"blk.64.attn_q.weight",                     "2",  5120,  12288,  35389440},
        {"blk.64.attn_q_norm.weight",                "0",   256,      0,      1024},
        {"blk.64.attn_v.weight",                     "2",  5120,   1024,   2949120},
        {"blk.64.ffn_down.weight",                   "2", 17408,   5120,  50135040},
        {"blk.64.ffn_gate.weight",                   "2",  5120,  17408,  50135040},
        {"blk.64.ffn_up.weight",                     "2",  5120,  17408,  50135040},
        {"blk.64.nextn.eh_proj.weight",              "2", 10240,   5120,  29491200},
        {"blk.64.nextn.enorm.weight",                "0",  5120,      0,     20480},
        {"blk.64.nextn.hnorm.weight",                "0",  5120,      0,     20480},
        {"blk.64.nextn.shared_head_norm.weight",     "0",  5120,      0,     20480},
        {"blk.64.post_attention_norm.weight",        "0",  5120,      0,     20480},
        {"blk.64.nextn.shared_head_head.weight",     "2",  5120,  40960, 117964800},
    };
    if (tensors.size() != 1 && tensors.size() != sizeof(EXPECTED) / sizeof(EXPECTED[0])) {
        error = "target manifest must contain exactly 1 or 17 tensors";
        return false;
    }
    const size_t count = tensors.size() == 1 ? 1 : sizeof(EXPECTED) / sizeof(EXPECTED[0]);
    for (size_t i = 0; i < count; ++i) {
        const Exp & item = EXPECTED[i];
        const Tens * tensor = bridgespec_artifact::find_tensor(tensors, item.name);
        const bool matches = tensor != nullptr && (item.ne1 == 0
            ? bridgespec_artifact::tensor_matches(*tensor, item.dtype, {item.ne0}, item.nbytes)
            : bridgespec_artifact::tensor_matches(*tensor, item.dtype, {item.ne0, item.ne1}, item.nbytes));
        if (!matches) {
            error = std::string("target tensor schema mismatch: ") + item.name;
            return false;
        }
    }
    return true;
}

DF_API int spec_dflash_init(const char * dir) {
    if (S.ok) return 0;
    char pb[1024];
    std::string validation_error;
    if (!bridgespec_artifact::make_path(pb, sizeof(pb), dir, "dflash_manifest.json") ||
        !bridgespec_artifact::load_manifest(pb, S.G, validation_error)) {
        fprintf(stderr, "[dflash-sidecar] invalid DFlash manifest: %s\n", validation_error.c_str());
        return -1;
    }
    if (!validate_dflash_schema(S.G, validation_error)) {
        fprintf(stderr, "[dflash-sidecar] %s\n", validation_error.c_str());
        return -1;
    }
    if (!bridgespec_artifact::make_path(pb, sizeof(pb), dir, "dflash_weights.bin")) return -1;
    FILE * wf = fopen(pb, "rb");
    uint64_t wf_size = 0;
    if (!wf || !bridgespec_artifact::file_size(wf, wf_size) ||
        !bridgespec_artifact::validate_blob_layout(S.G, wf_size, validation_error)) {
        fprintf(stderr, "[dflash-sidecar] invalid DFlash weights layout: %s\n", validation_error.c_str());
        if (wf) fclose(wf);
        return -1;
    }

    // Validate the complete target embedding blob, not only the descriptor used
    // by this sidecar. prepare_assets emits either the embedding alone or the
    // complete 17-tensor MTP blob.
    std::vector<Tens> target_manifest;
    if (!bridgespec_artifact::make_path(pb, sizeof(pb), dir, "drafter_manifest.json") ||
        !bridgespec_artifact::load_manifest(pb, target_manifest, validation_error) ||
        !validate_target_schema(target_manifest, validation_error)) {
        fprintf(stderr, "[dflash-sidecar] invalid target manifest: %s\n", validation_error.c_str());
        fclose(wf);
        return -1;
    }
    const Tens * token_embedding = bridgespec_artifact::find_tensor(target_manifest, "token_embd.weight");
    if (!bridgespec_artifact::make_path(pb, sizeof(pb), dir, "drafter_weights.bin")) { fclose(wf); return -1; }
    FILE * target_wf = fopen(pb, "rb");
    uint64_t target_wf_size = 0;
    if (!target_wf || !bridgespec_artifact::file_size(target_wf, target_wf_size) ||
        !bridgespec_artifact::validate_blob_layout(target_manifest, target_wf_size, validation_error)) {
        fprintf(stderr, "[dflash-sidecar] invalid target weights layout: %s\n", validation_error.c_str());
        if (target_wf) fclose(target_wf);
        fclose(wf);
        return -1;
    }

    // Validate the selected target head and its host-side remap before any HIP
    // allocation/upload. This keeps malformed artifacts out of GPU read paths.
    const bool full_head = getenv("LLAMA_SPEC_HIP_FULL_HEAD") != nullptr;
    const int expected_rows = full_head ? DF_VOCAB : 40960;
    const uint64_t expected_head_size = static_cast<uint64_t>(expected_rows) * 4200;
    if (!bridgespec_artifact::make_path(pb, sizeof(pb), dir, full_head ? "target_head.bin" : "target_head_sliced.bin")) {
        fclose(target_wf); fclose(wf); return -1;
    }
    FILE * head_file = fopen(pb, "rb");
    uint64_t head_size = 0;
    if (!head_file || !bridgespec_artifact::file_size(head_file, head_size) || head_size != expected_head_size) {
        fprintf(stderr, "[dflash-sidecar] target-head size mismatch: expected %llu bytes\n",
                static_cast<unsigned long long>(expected_head_size));
        if (head_file) fclose(head_file);
        fclose(target_wf); fclose(wf);
        return -1;
    }
    S.n_head = expected_rows;
    std::vector<int32_t> remap_h(static_cast<size_t>(S.n_head));
    if (full_head) {
        for (int i = 0; i < S.n_head; ++i) remap_h[i] = i;
    } else {
        if (!bridgespec_artifact::make_path(pb, sizeof(pb), dir, "draft_head_ids.bin")) {
            fclose(head_file); fclose(target_wf); fclose(wf); return -1;
        }
        FILE * remap_file = fopen(pb, "rb");
        uint64_t remap_size = 0;
        const uint64_t expected_remap_size = static_cast<uint64_t>(S.n_head) * sizeof(int32_t);
        if (!remap_file || !bridgespec_artifact::file_size(remap_file, remap_size) ||
            remap_size != expected_remap_size ||
            !bridgespec_artifact::read_exact_at(remap_file, 0, remap_h.data(), expected_remap_size)) {
            fprintf(stderr, "[dflash-sidecar] remap size/read mismatch\n");
            if (remap_file) fclose(remap_file);
            fclose(head_file); fclose(target_wf); fclose(wf);
            return -1;
        }
        fclose(remap_file);
        if (!bridgespec_artifact::validate_remap(remap_h, DF_VOCAB, validation_error)) {
            fprintf(stderr, "[dflash-sidecar] invalid remap: %s\n", validation_error.c_str());
            fclose(head_file); fclose(target_wf); fclose(wf);
            return -1;
        }
    }

    // All external metadata/ranges are now validated. The normal upload path is
    // unchanged from here onward and runtime kernels do no manifest work.
    std::vector<char> hb;
    for (auto & t : S.G) {
        hb.resize(static_cast<size_t>(t.nbytes));
        if (!bridgespec_artifact::read_exact_at(wf, t.offset, hb.data(), t.nbytes)) { fclose(wf); return -1; }
        DFC(hipMalloc(&t.dev, static_cast<size_t>(t.nbytes)));
        DFC(hipMemcpy(t.dev, hb.data(), static_cast<size_t>(t.nbytes), hipMemcpyHostToDevice));
    }
    // selector tables -> host f32
    S.sel_prev.resize((size_t) DF_VOCAB * DF_RANK);
    S.sel_next.resize((size_t) DF_VOCAB * DF_RANK);
    for (auto & pr : { std::make_pair("selector_predecessor.weight", S.sel_prev.data()),
                       std::make_pair("selector_successor.weight",   S.sel_next.data()) }) {
        Tens * t = DT(pr.first);
        if (!t) { fclose(wf); return -1; }
        hb.resize(static_cast<size_t>(t->nbytes));
        if (!bridgespec_artifact::read_exact_at(wf, t->offset, hb.data(), t->nbytes)) { fclose(wf); return -1; }
        deq_q4k_host((const uint8_t *) hb.data(), pr.second, (size_t) DF_VOCAB * DF_RANK);
    }
    fclose(wf);

    // target token_embd pinned host (from drafter_weights.bin)
    {
        void * hptr = nullptr;
        DFC(hipHostMalloc(&hptr, static_cast<size_t>(token_embedding->nbytes), hipHostMallocMapped));
        if (!bridgespec_artifact::read_exact_at(target_wf, token_embedding->offset, hptr, token_embedding->nbytes)) {
            fclose(target_wf); return -1;
        }
        fclose(target_wf);
        void * dp = nullptr;
        DFC(hipHostGetDevicePointer(&dp, hptr, 0));
        S.W_embd = (const block_q4_0d *) dp;
    }
    // target head: sliced 40960 rows + id remap by default; full head uses identity
    {
        std::vector<char> buf(static_cast<size_t>(head_size));
        if (!bridgespec_artifact::read_exact_at(head_file, 0, buf.data(), head_size)) { fclose(head_file); return -1; }
        fclose(head_file);
        void * d = nullptr;
        DFC(hipMalloc(&d, static_cast<size_t>(head_size)));
        DFC(hipMemcpy(d, buf.data(), static_cast<size_t>(head_size), hipMemcpyHostToDevice));
        S.W_head = (const block_q6_K *) d;
        fprintf(stderr, "[dflash-sidecar] head: %s (%d rows)\n", full_head ? "FULL" : "sliced", S.n_head);
    }
    S.remap = std::move(remap_h);
    if (!full_head) {
        fprintf(stderr, "[dflash-sidecar] release-abi: %d bundled-remap\n", DF_RELEASE_ABI);
    }
    for (int il = 0; il < DF_LAYERS; ++il) {
        auto nm = [&](const char * s2) { char b2[128]; snprintf(b2, sizeof(b2), "blk.%d.%s", il, s2); return std::string(b2); };
        auto req = [&](const std::string & n) -> Tens * { Tens * t = DT(n); if (!t) { fprintf(stderr, "[dflash-sidecar] missing %s\n", n.c_str()); } return t; };
        Tens *t;
        if (!(t = req(nm("attn_conv_proj.weight")))) return -1; S.L[il].conv_a_proj = (const block_q4_K *) t->dev;
        if (!(t = req(nm("ffn_conv_proj.weight"))))  return -1; S.L[il].conv_f_proj = (const block_q4_K *) t->dev;
        if (!(t = req(nm("attn_q.weight")))) return -1; S.L[il].wq = (const block_q4_K *) t->dev;
        if (!(t = req(nm("attn_k.weight")))) return -1; S.L[il].wk = (const block_q4_K *) t->dev;
        if (!(t = req(nm("attn_v.weight")))) return -1; S.L[il].wv = t->dev; S.L[il].wv_q6 = t->dtype == "14";
        if (!(t = req(nm("attn_output.weight")))) return -1; S.L[il].wo = (const block_q4_K *) t->dev;
        if (!(t = req(nm("ffn_up.weight"))))   return -1; S.L[il].up   = (const block_q4_K *) t->dev;
        if (!(t = req(nm("ffn_gate.weight")))) return -1; S.L[il].gate = (const block_q4_K *) t->dev;
        if (!(t = req(nm("ffn_down.weight")))) return -1; S.L[il].down = t->dev; S.L[il].down_q6 = t->dtype == "14";
        if (!(t = req(nm("attn_norm.weight")))) return -1; S.L[il].attn_norm = (const float *) t->dev;
        if (!(t = req(nm("attn_q_norm.weight")))) return -1; S.L[il].qn = (const float *) t->dev;
        if (!(t = req(nm("attn_k_norm.weight")))) return -1; S.L[il].kn = (const float *) t->dev;
        if (!(t = req(nm("ffn_norm.weight")))) return -1; S.L[il].ffn_norm = (const float *) t->dev;
        if (!(t = req(nm("attn_conv_base")))) return -1; S.L[il].conv_a_base = (const float *) t->dev;
        if (!(t = req(nm("ffn_conv_base")))) return -1; S.L[il].conv_f_base = (const float *) t->dev;
    }
    Tens * t;
    if (!(t = DT("fc.weight"))) return -1; S.W_fc = (const block_q4_K *) t->dev;
    if (!(t = DT("enc.output_norm.weight"))) return -1; S.w_enc_norm = (const float *) t->dev;
    if (!(t = DT("output_norm.weight"))) return -1; S.w_out_norm = (const float *) t->dev;
    if (!(t = DT("selector_hidden.weight"))) return -1; S.W_selhid = (const block_q4_K *) t->dev;

    if (const char * mp = getenv("LLAMA_SPEC_HIP_MAX_POS")) {
        long long v = atoll(mp);
        if (v < 4096) v = 4096;
        if (v > 131072) v = 131072;
        S.max_pos = (int) v;
    }
    DFC(hipStreamCreate(&S.st));
    DFC(hipMalloc(&S.feat_d, (size_t) 8 * 25600 * 4));
    DFC(hipMalloc(&S.henc,   (size_t) 8 * DF_EMBD * 4));
    DFC(hipMalloc(&S.x,      (size_t) DF_BLOCK * DF_EMBD * 4));
    DFC(hipMalloc(&S.xn,     (size_t) DF_BLOCK * DF_EMBD * 4));
    DFC(hipMalloc(&S.dyn_a,  (size_t) DF_BLOCK * DF_CONVDYN * 4));
    DFC(hipMalloc(&S.dyn_f,  (size_t) DF_BLOCK * DF_CONVDYN * 4));
    DFC(hipMalloc(&S.cg,     (size_t) DF_BLOCK * DF_EMBD * 4));
    DFC(hipMalloc(&S.q,      (size_t) DF_BLOCK * DF_HEADS * DF_HDIM * 4));
    DFC(hipMalloc(&S.k,      (size_t) 8 * DF_KV * DF_HDIM * 4));
    DFC(hipMalloc(&S.v,      (size_t) 8 * DF_KV * DF_HDIM * 4));
    DFC(hipMalloc(&S.att,    (size_t) DF_BLOCK * DF_HEADS * DF_HDIM * 4));
    DFC(hipMalloc(&S.wo_out, (size_t) DF_BLOCK * DF_EMBD * 4));
    DFC(hipMalloc(&S.fu,     (size_t) DF_BLOCK * DF_FF * 4));
    DFC(hipMalloc(&S.fg,     (size_t) DF_BLOCK * DF_FF * 4));
    DFC(hipMalloc(&S.ftmp,   (size_t) DF_BLOCK * DF_EMBD * 4));
    DFC(hipMalloc(&S.resid,  (size_t) DF_BLOCK * DF_EMBD * 4));
    DFC(hipMalloc(&S.tE,     (size_t) DF_BLOCK * DF_EMBD * 4));
    DFC(hipMalloc(&S.logits, (size_t) DF_BLOCK * S.n_head * 4));
    DFC(hipMalloc(&S.selhid, (size_t) DF_BLOCK * DF_RANK * 4));
    DFC(hipMalloc(&S.pos_d, 8 * 4));
    DFC(hipMalloc(&S.tok_d, 8 * 4));
    S.tkparts = (S.n_head + 2047) / 2048;
    DFC(hipMalloc(&S.tk_pmax, (size_t) 8 * S.tkparts * 4));
    DFC(hipMalloc(&S.tk_pidx, (size_t) 8 * S.tkparts * 4));
    DFC(hipMalloc(&S.tk_ids,  8 * DF_TOPK * 4));
    DFC(hipMalloc(&S.tk_vals, 8 * DF_TOPK * 4));
    for (int il = 0; il < DF_LAYERS; ++il) {
        DFC(hipMalloc(&S.Kc[il], (size_t) S.max_pos * DF_KV * DF_HDIM * 2));
        DFC(hipMalloc(&S.Vc[il], (size_t) S.max_pos * DF_KV * DF_HDIM * 2));
    }
    S.ok = true;
    fprintf(stderr, "[dflash-sidecar] initialized (max_pos=%d)\n", S.max_pos);
    return 0;
}

// encoder + KV inject for up to n rows of target features
DF_API int spec_dflash_chunk(const int32_t * poss, const float * feats, int n) {
    if (!S.ok || n < 0 || !poss || !feats) return -1;
    static int n_calls = 0, n_rows = 0;
    static double tot_ms = 0.0;
    const bool dbg = getenv("LLAMA_SPEC_HIP_DEBUG") != nullptr;
    auto t0 = std::chrono::high_resolution_clock::now();
    const int NROT = 128;
    for (int off = 0; off < n; off += 8) {
        const int nb = std::min(8, n - off);
        bool skip = false;
        for (int i = 0; i < nb; ++i) if (poss[off + i] < 0 || poss[off + i] >= S.max_pos) skip = true;
        if (skip) continue;
        DFC(hipMemcpyAsync(S.feat_d, feats + (size_t) off * 25600, (size_t) nb * 25600 * 4, hipMemcpyHostToDevice, S.st));
        DFC(hipMemcpyAsync(S.pos_d, poss + off, nb * 4, hipMemcpyHostToDevice, S.st));
        DISPATCH_Q4K(nb, S.W_fc, S.feat_d, S.henc, 25600, DF_EMBD, S.st);
        df_rmsnorm<<<nb, 256, 0, S.st>>>(S.henc, S.w_enc_norm, S.xn, DF_EMBD);
        for (int il = 0; il < DF_LAYERS; ++il) {
            DISPATCH_Q4K(nb, S.L[il].wk, S.xn, S.k, DF_EMBD, DF_KV * DF_HDIM, S.st);
            if (S.L[il].wv_q6) DISPATCH_Q6K(nb, (const block_q6_K *) S.L[il].wv, S.xn, S.v, DF_EMBD, DF_KV * DF_HDIM, S.st);
            else               DISPATCH_Q4K(nb, (const block_q4_K *) S.L[il].wv, S.xn, S.v, DF_EMBD, DF_KV * DF_HDIM, S.st);
            df_rmsnorm_head<<<dim3(DF_KV, nb), 128, 0, S.st>>>(S.k, S.L[il].kn, DF_KV);
            df_rope<<<dim3(DF_KV, nb), NROT / 2, 0, S.st>>>(S.k, DF_KV, S.pos_d, NROT);
            df_kv_store<<<dim3((DF_KV * DF_HDIM + 255) / 256, nb), 256, 0, S.st>>>(S.k, S.v, S.Kc[il], S.Vc[il], S.pos_d, nb);
        }
        DFC(hipStreamSynchronize(S.st));
    }
    if (dbg) {
        auto t1 = std::chrono::high_resolution_clock::now();
        tot_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        n_rows += n;
        if (++n_calls % 25 == 0) {
            fprintf(stderr, "[dflash-sidecar] chunk avg %.2f ms (%.1f rows) over %d calls\n",
                    tot_ms / n_calls, (double) n_rows / n_calls, n_calls);
        }
    }
    return 0;
}

DF_API int spec_dflash_draft(int32_t id_last, int32_t n_past, int32_t * out_ids) {
    if (!S.ok || !out_ids || id_last < 0 || id_last >= DF_VOCAB) return -1;
    if (n_past < 1 || static_cast<uint64_t>(n_past) + DF_BLOCK > static_cast<uint64_t>(S.max_pos)) return -1;
    const int NROT = 128;
    static int n_calls = 0;
    static double tot_ms = 0.0;
    const bool dbg = getenv("LLAMA_SPEC_HIP_DEBUG") != nullptr;
    auto t0 = std::chrono::high_resolution_clock::now();

    int toks[8], poss[8];
    toks[0] = id_last;
    for (int i = 1; i < 8; ++i) toks[i] = 248070;
    for (int i = 0; i < 8; ++i) poss[i] = n_past + i;
    DFC(hipMemcpyAsync(S.tok_d, toks, 32, hipMemcpyHostToDevice, S.st));
    DFC(hipMemcpyAsync(S.pos_d, poss, 32, hipMemcpyHostToDevice, S.st));

    // one hipGraph for the whole ~205-kernel draft: all kernel args are static
    // device pointers (contents vary per call), n_past is read from pos_d[0]
    static hipGraphExec_t gexec = nullptr;
    if (gexec != nullptr) {
        DFC(hipGraphLaunch(gexec, S.st));
        goto readback;
    }
    DFC(hipStreamBeginCapture(S.st, hipStreamCaptureModeThreadLocal));
    df_embed_rows<<<dim3((DF_EMBD + 255) / 256, DF_BLOCK), 256, 0, S.st>>>(S.W_embd, S.tok_d, S.x, DF_EMBD);
    for (int il = 0; il < DF_LAYERS; ++il) {
        auto & L = S.L[il];
        df_rmsnorm<<<DF_BLOCK, 256, 0, S.st>>>(S.x, L.attn_norm, S.xn, DF_EMBD);
        DISPATCH_Q4K(DF_BLOCK, L.conv_a_proj, S.xn, S.dyn_a, DF_EMBD, DF_CONVDYN, S.st);
        df_conv_gate<<<dim3((DF_EMBD + 255) / 256, DF_BLOCK), 256, 0, S.st>>>(S.xn, S.dyn_a, L.conv_a_base, S.cg, 0, DF_BLOCK);
        DISPATCH_Q4K(DF_BLOCK, L.wq, S.cg, S.q, DF_EMBD, DF_HEADS * DF_HDIM, S.st);
        DISPATCH_Q4K(DF_BLOCK, L.wk, S.cg, S.k, DF_EMBD, DF_KV * DF_HDIM, S.st);
        if (L.wv_q6) DISPATCH_Q6K(DF_BLOCK, (const block_q6_K *) L.wv, S.cg, S.v, DF_EMBD, DF_KV * DF_HDIM, S.st);
        else         DISPATCH_Q4K(DF_BLOCK, (const block_q4_K *) L.wv, S.cg, S.v, DF_EMBD, DF_KV * DF_HDIM, S.st);
        df_rmsnorm_head<<<dim3(DF_HEADS, DF_BLOCK), 128, 0, S.st>>>(S.q, L.qn, DF_HEADS);
        df_rmsnorm_head<<<dim3(DF_KV, DF_BLOCK), 128, 0, S.st>>>(S.k, L.kn, DF_KV);
        df_rope<<<dim3(DF_HEADS, DF_BLOCK), NROT / 2, 0, S.st>>>(S.q, DF_HEADS, S.pos_d, NROT);
        df_rope<<<dim3(DF_KV, DF_BLOCK), NROT / 2, 0, S.st>>>(S.k, DF_KV, S.pos_d, NROT);
        df_attn<<<dim3(DF_HEADS, DF_BLOCK), 256, 0, S.st>>>(S.q, S.Kc[il], S.Vc[il], S.k, S.v, S.att, n_past, S.pos_d);
        DISPATCH_Q4K(DF_BLOCK, L.wo, S.att, S.wo_out, DF_HEADS * DF_HDIM, DF_EMBD, S.st);
        df_conv_gate<<<dim3((DF_EMBD + 255) / 256, DF_BLOCK), 256, 0, S.st>>>(S.wo_out, S.dyn_a, L.conv_a_base, S.cg, 1, DF_BLOCK);
        df_add<<<(DF_BLOCK * DF_EMBD + 255) / 256, 256, 0, S.st>>>(S.x, S.cg, DF_BLOCK * DF_EMBD);
        df_rmsnorm<<<DF_BLOCK, 256, 0, S.st>>>(S.x, L.ffn_norm, S.xn, DF_EMBD);
        DISPATCH_Q4K(DF_BLOCK, L.conv_f_proj, S.xn, S.dyn_f, DF_EMBD, DF_CONVDYN, S.st);
        df_conv_gate<<<dim3((DF_EMBD + 255) / 256, DF_BLOCK), 256, 0, S.st>>>(S.xn, S.dyn_f, L.conv_f_base, S.cg, 0, DF_BLOCK);
        DISPATCH_Q4K(DF_BLOCK, L.up, S.cg, S.fu, DF_EMBD, DF_FF, S.st);
        DISPATCH_Q4K(DF_BLOCK, L.gate, S.cg, S.fg, DF_EMBD, DF_FF, S.st);
        df_silu_mul<<<(DF_BLOCK * DF_FF + 255) / 256, 256, 0, S.st>>>(S.fg, S.fu, DF_BLOCK * DF_FF);
        if (L.down_q6) DISPATCH_Q6K(DF_BLOCK, (const block_q6_K *) L.down, S.fg, S.ftmp, DF_FF, DF_EMBD, S.st);
        else           DISPATCH_Q4K(DF_BLOCK, (const block_q4_K *) L.down, S.fg, S.ftmp, DF_FF, DF_EMBD, S.st);
        df_conv_gate<<<dim3((DF_EMBD + 255) / 256, DF_BLOCK), 256, 0, S.st>>>(S.ftmp, S.dyn_f, L.conv_f_base, S.cg, 1, DF_BLOCK);
        df_add<<<(DF_BLOCK * DF_EMBD + 255) / 256, 256, 0, S.st>>>(S.x, S.cg, DF_BLOCK * DF_EMBD);
    }
    df_rmsnorm<<<DF_BLOCK, 256, 0, S.st>>>(S.x, S.w_out_norm, S.tE, DF_EMBD);
    gemv8t_q6_K<DF_BLOCK><<<(S.n_head + 7) / 8, 256, 0, S.st>>>(S.W_head, S.tE, S.logits, DF_EMBD, S.n_head);
    DISPATCH_Q4K(DF_BLOCK, S.W_selhid, S.tE, S.selhid, DF_EMBD, DF_RANK, S.st);
    for (int s2 = 0; s2 < DF_TOPK; ++s2) {
        dfa_argmax_partial<<<dim3(S.tkparts, DF_BLOCK), 256, 0, S.st>>>(S.logits, S.n_head, S.tk_pmax, S.tk_pidx, S.tkparts);
        dfa_argmax_final_mask<<<DF_BLOCK, 1, 0, S.st>>>(S.logits, S.n_head, S.tk_pmax, S.tk_pidx, S.tkparts, S.tk_ids, S.tk_vals, s2, DF_TOPK);
    }
    {
        hipGraph_t graph = nullptr;
        DFC(hipStreamEndCapture(S.st, &graph));
        DFC(hipGraphInstantiate(&gexec, graph, nullptr, nullptr, 0));
        DFC(hipGraphDestroy(graph));
        fprintf(stderr, "[dflash-sidecar] draft graph captured\n");
        DFC(hipGraphLaunch(gexec, S.st));
    }
readback:
    int   h_ids[8 * DF_TOPK];
    float h_vals[8 * DF_TOPK];
    float sh[8 * DF_RANK];
    DFC(hipMemcpyAsync(h_ids, S.tk_ids, sizeof(h_ids), hipMemcpyDeviceToHost, S.st));
    DFC(hipMemcpyAsync(h_vals, S.tk_vals, sizeof(h_vals), hipMemcpyDeviceToHost, S.st));
    DFC(hipMemcpyAsync(sh, S.selhid, sizeof(sh), hipMemcpyDeviceToHost, S.st));
    DFC(hipStreamSynchronize(S.st));

    int pred_idx = 0;
    int n_out = 0;
    for (int t = 1; t < 8; ++t) {
        const float * hid = sh + (size_t) t * DF_RANK;
        const int ps = (t == 1) ? -1 : h_ids[(t - 1) * DF_TOPK + pred_idx];
        const int pred_id = (t == 1) ? id_last : ((ps >= 0 && ps < S.n_head) ? S.remap[ps] : -1);
        if (pred_id < 0 || pred_id >= DF_VOCAB) break;
        const float * pv = S.sel_prev.data() + (size_t) pred_id * DF_RANK;
        float best = -1e30f; int besti = -1;
        for (int kx = 0; kx < DF_TOPK; ++kx) {
            const int cs = h_ids[t * DF_TOPK + kx];
            if (cs < 0 || cs >= S.n_head) continue;
            const int cid = S.remap[cs];
            if (cid < 0 || cid >= DF_VOCAB) continue;
            const float * nx = S.sel_next.data() + (size_t) cid * DF_RANK;
            float s2 = 0.0f;
            for (int rr = 0; rr < DF_RANK; ++rr) s2 += nx[rr] * (pv[rr] * hid[rr]);
            s2 += h_vals[t * DF_TOPK + kx];
            if (s2 == s2 && (besti < 0 || s2 > best)) { best = s2; besti = kx; }
        }
        if (besti < 0) break;
        pred_idx = besti;
        const int ts2 = h_ids[t * DF_TOPK + besti];
        if (ts2 < 0 || ts2 >= S.n_head) break;
        const int tokid = S.remap[ts2];
        if (tokid < 0 || tokid >= DF_VOCAB) break;
        out_ids[n_out++] = tokid;
    }
    if (dbg) {
        auto t1 = std::chrono::high_resolution_clock::now();
        tot_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (++n_calls % 25 == 0) {
            fprintf(stderr, "[dflash-sidecar] draft avg %.2f ms over %d calls\n", tot_ms / n_calls, n_calls);
        }
    }
    return n_out;
}
