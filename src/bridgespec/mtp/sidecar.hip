// SPDX-License-Identifier: MIT
// HIP draft sidecar: C API around the MTP-block kernels, built as a DLL that
// llama.cpp's speculative driver loads at runtime (LLAMA_SPEC_HIP_SIDECAR).
// The harness build (-DSIDECAR_HARNESS) replays golden.bin through the SAME API.
#include "kernels.hip"
#include "kernels_b.hip"
#include "../artifact_manifest.h"

// ---------------- host: weights ----------------
using Tensor = bridgespec_artifact::TensorDesc;
static constexpr int SC_N_VOCAB = 248320;

// ---------------- sidecar state (single instance) ----------------
namespace {
#define N_SCR 4
struct Scratch {
    float *x_tok, *x_h, *e_n, *h_n, *cat, *proj, *attin, *qfull, *q, *gate, *k, *v, *attout,
          *ffn_in, *ffn_g, *ffn_u, *ffn_tmp, *resid, *hnorm2;
    int * id_dev;
};
#define N_BATCH 8
struct BatchScratch { // packed [N_BATCH][...] buffers for batched catch-up
    float *x_tok, *e_n, *h_n, *cat, *proj, *attin, *qfull, *q, *gate, *k, *v, *attout,
          *ffn_in, *ffn_g, *ffn_u, *ffn_tmp, *resid;
    int *toks, *poss;
};
struct State {
    bool ok = false;
    std::vector<Tensor> tensors;
    hipStream_t st;          // main (draft) stream
    hipStream_t rst[N_SCR];  // catch-up row streams
    Scratch scr[N_SCR];      // per-stream scratch; scr[0] doubles as the draft scratch
    BatchScratch bs;
    float *part_m, *part_l, *part_acc; // split-K attention partials (draft path)
    float *logits, *pmax;
    __half *Kc, *Vc;
    int *pidx, *tok_out;
    int nparts;
    size_t max_pos = 131072;
    hipEvent_t ev_a, ev_b; // GPU-side timing (debug)
    // weights
    const block_q4_0 *W_embd, *W_ehp, *W_q, *W_k, *W_v, *W_o, *W_fg, *W_fu, *W_fd, *W_head;
    const float *w_enorm, *w_hnorm, *w_attn, *w_qn, *w_kn, *w_post, *w_shn;
    int * remap_d = nullptr;
};
State S;

Tensor * find_tensor(const char * name) {
    return bridgespec_artifact::find_tensor(S.tensors, name);
}

#define SC_CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "[hip-sidecar] %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); return -1; } } while (0)

// phase 1: inputs -> projection -> Q/K/V -> rope -> KV store (rows independent)
void block_phase1(Scratch & c, hipStream_t st, const float * xh_in, int pos) {
    auto & s = S;
    rmsnorm<<<1, 256, 0, st>>>(c.x_tok, s.w_enorm, c.e_n, N_EMBD);
    rmsnorm<<<1, 256, 0, st>>>(xh_in,   s.w_hnorm, c.h_n, N_EMBD);
    concat2<<<(2 * N_EMBD + 255) / 256, 256, 0, st>>>(c.e_n, c.h_n, c.cat, N_EMBD);
    gemv_q4_0<<<N_EMBD, 32, 0, st>>>(s.W_ehp, c.cat, c.proj, 2 * N_EMBD, N_EMBD);
    rmsnorm<<<1, 256, 0, st>>>(c.proj, s.w_attn, c.attin, N_EMBD);
    gemv_q4_0<<<2 * N_HEAD * HEAD_DIM, 32, 0, st>>>(s.W_q, c.attin, c.qfull, N_EMBD, 2 * N_HEAD * HEAD_DIM);
    gemv_q4_0<<<N_HEAD_KV * HEAD_DIM, 32, 0, st>>>(s.W_k, c.attin, c.k, N_EMBD, N_HEAD_KV * HEAD_DIM);
    gemv_q4_0<<<N_HEAD_KV * HEAD_DIM, 32, 0, st>>>(s.W_v, c.attin, c.v, N_EMBD, N_HEAD_KV * HEAD_DIM);
    split_q_gate<<<N_HEAD, HEAD_DIM, 0, st>>>(c.qfull, c.q, c.gate);
    rmsnorm_heads<<<N_HEAD, 256, 0, st>>>(c.q, s.w_qn, c.q, N_HEAD);
    rmsnorm_heads<<<N_HEAD_KV, 256, 0, st>>>(c.k, s.w_kn, c.k, N_HEAD_KV);
    rope_neox<<<N_HEAD, N_ROT / 2, 0, st>>>(c.q, N_HEAD, pos);
    rope_neox<<<N_HEAD_KV, N_ROT / 2, 0, st>>>(c.k, N_HEAD_KV, pos);
    kv_store<<<(N_HEAD_KV * HEAD_DIM + 255) / 256, 256, 0, st>>>(c.k, c.v, s.Kc, s.Vc, pos);
}

// phase 2: attention (reads global KV, all rows' stores must be complete) -> ffn
void block_phase2(Scratch & c, hipStream_t st, int pos, int * tok_dst) {
    auto & s = S;
    // split-K attention (winner of v3/v4/v5 shootout): workgroup per (q-head,
    // 4096-position chunk). The GQA shared-K variants lost to LDS pressure.
    const int att_parts = (pos + 1 + ATTN_SPLIT - 1) / ATTN_SPLIT;
    attn_1q_split<<<dim3(N_HEAD, att_parts), 256, 0, st>>>(c.q, s.Kc, s.Vc, pos + 1, s.part_m, s.part_l, s.part_acc);
    attn_merge<<<N_HEAD, HEAD_DIM, 0, st>>>(s.part_m, s.part_l, s.part_acc, c.attout, att_parts);
    sigmoid_gate<<<(N_HEAD * HEAD_DIM + 255) / 256, 256, 0, st>>>(c.attout, c.gate, N_HEAD * HEAD_DIM);
    gemv_q4_0<<<N_EMBD, 32, 0, st>>>(s.W_o, c.attout, c.resid, N_HEAD * HEAD_DIM, N_EMBD);
    add_inplace<<<(N_EMBD + 255) / 256, 256, 0, st>>>(c.resid, c.proj, N_EMBD);
    rmsnorm<<<1, 256, 0, st>>>(c.resid, s.w_post, c.ffn_in, N_EMBD);
    gemv_q4_0<<<N_FF, 32, 0, st>>>(s.W_fg, c.ffn_in, c.ffn_g, N_EMBD, N_FF);
    gemv_q4_0<<<N_FF, 32, 0, st>>>(s.W_fu, c.ffn_in, c.ffn_u, N_EMBD, N_FF);
    silu_mul<<<(N_FF + 255) / 256, 256, 0, st>>>(c.ffn_g, c.ffn_u, N_FF);
    gemv_q4_0<<<N_EMBD, 32, 0, st>>>(s.W_fd, c.ffn_g, c.ffn_tmp, N_FF, N_EMBD);
    add_inplace<<<(N_EMBD + 255) / 256, 256, 0, st>>>(c.resid, c.ffn_tmp, N_EMBD);
    if (tok_dst != nullptr) {
        rmsnorm<<<1, 256, 0, st>>>(c.resid, s.w_shn, c.hnorm2, N_EMBD);
        gemv_q4_0<<<N_HEAD_ROWS, 32, 0, st>>>(s.W_head, c.hnorm2, s.logits, N_EMBD, N_HEAD_ROWS);
        argmax_partial<<<s.nparts, 256, 0, st>>>(s.logits, N_HEAD_ROWS, s.pmax, s.pidx);
        argmax_final<<<1, 1, 0, st>>>(s.pmax, s.pidx, s.nparts, N_HEAD_ROWS,
                                      s.remap_d, SC_N_VOCAB, tok_dst);
    }
}
} // namespace

#ifdef _WIN32
#define SC_API extern "C" __declspec(dllexport)
#else
#define SC_API extern "C"
#endif

static constexpr int32_t SC_RELEASE_ABI = 1;

SC_API int spec_hip_release_abi() {
    return SC_RELEASE_ABI;
}

// ABI/model handshake: the host passes what it believes the model is; refuse
// to run against a mismatched drafter (wrong arch/width/head slice).
SC_API int spec_hip_check(int32_t n_embd, int32_t head_rows) {
    return (n_embd == N_EMBD && head_rows == N_HEAD_ROWS) ? 0 : -1;
}

// init: weights_dir holds drafter_manifest.json + drafter_weights.bin;
// ids_path = little-endian i32 slice->full-vocab remap table (40960 entries).
SC_API int spec_hip_init(const char * weights_dir, const char * ids_path) {
    if (S.ok) return 0;
    char pbuf[1024];
    std::string validation_error;
    if (!bridgespec_artifact::make_path(pbuf, sizeof(pbuf), weights_dir, "drafter_manifest.json")) {
        fprintf(stderr, "[hip-sidecar] invalid or overlong weights directory\n");
        return -1;
    }
    if (!bridgespec_artifact::load_manifest(pbuf, S.tensors, validation_error)) {
        fprintf(stderr, "[hip-sidecar] invalid manifest %s: %s\n", pbuf, validation_error.c_str());
        return -1;
    }

    // model-identity validation: exact expected tensor set for the Qwen3.8-27B
    // nextn drafter. Any name/dtype/shape/size mismatch refuses init.
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
    if (S.tensors.size() != sizeof(EXPECTED)/sizeof(EXPECTED[0])) {
        fprintf(stderr, "[hip-sidecar] manifest has %zu tensors, expected %zu\n", S.tensors.size(), sizeof(EXPECTED)/sizeof(EXPECTED[0]));
        return -1;
    }
    for (const auto & e : EXPECTED) {
        Tensor * t = find_tensor(e.name);
        if (!t) { fprintf(stderr, "[hip-sidecar] missing tensor %s\n", e.name); return -1; }
        const bool matches = e.ne1 == 0
            ? bridgespec_artifact::tensor_matches(*t, e.dtype, {e.ne0}, e.nbytes)
            : bridgespec_artifact::tensor_matches(*t, e.dtype, {e.ne0, e.ne1}, e.nbytes);
        if (!matches) {
            fprintf(stderr, "[hip-sidecar] tensor schema mismatch: %s\n", e.name);
            return -1;
        }
    }

    if (!bridgespec_artifact::make_path(pbuf, sizeof(pbuf), weights_dir, "drafter_weights.bin")) return -1;
    FILE * wf = fopen(pbuf, "rb");
    if (!wf) { fprintf(stderr, "[hip-sidecar] cannot open %s\n", pbuf); return -1; }
    uint64_t weights_size = 0;
    if (!bridgespec_artifact::file_size(wf, weights_size) ||
        !bridgespec_artifact::validate_blob_layout(S.tensors, weights_size, validation_error)) {
        fprintf(stderr, "[hip-sidecar] invalid weights layout: %s\n", validation_error.c_str());
        fclose(wf);
        return -1;
    }

    // Validate the complete little-endian remap before any HIP allocation or
    // artifact upload. A valid sliced head is an injective subset of the target
    // vocabulary, exactly N_HEAD_ROWS entries long.
    std::vector<int32_t> remap_h(N_HEAD_ROWS);
    FILE * rf = ids_path == nullptr ? nullptr : fopen(ids_path, "rb");
    uint64_t remap_size = 0;
    const uint64_t expected_remap_size = static_cast<uint64_t>(N_HEAD_ROWS) * sizeof(int32_t);
    if (!rf || !bridgespec_artifact::file_size(rf, remap_size) || remap_size != expected_remap_size ||
        !bridgespec_artifact::read_exact_at(rf, 0, remap_h.data(), expected_remap_size)) {
        fprintf(stderr, "[hip-sidecar] invalid remap table (%s): expected %llu bytes\n",
                ids_path ? ids_path : "null", static_cast<unsigned long long>(expected_remap_size));
        if (rf) fclose(rf);
        fclose(wf);
        return -1;
    }
    fclose(rf);
    if (!bridgespec_artifact::validate_remap(remap_h, SC_N_VOCAB, validation_error)) {
        fprintf(stderr, "[hip-sidecar] invalid remap table: %s\n", validation_error.c_str());
        fclose(wf);
        return -1;
    }

    std::vector<char> hostbuf;
    for (auto & t : S.tensors) {
        // token_embd is only ever read one row (2.9KB) per draft step: keep it in
        // pinned host memory (zero VRAM) instead of adding 760MB of device pressure
        const bool host_resident = t.name == "token_embd.weight";
        if (host_resident) {
            void * hptr = nullptr;
            SC_CHECK(hipHostMalloc(&hptr, static_cast<size_t>(t.nbytes), hipHostMallocMapped));
            if (!bridgespec_artifact::read_exact_at(wf, t.offset, hptr, t.nbytes)) { fclose(wf); fprintf(stderr, "[hip-sidecar] short read %s\n", t.name.c_str()); return -1; }
            SC_CHECK(hipHostGetDevicePointer(&t.dev, hptr, 0));
        } else {
            hostbuf.resize(static_cast<size_t>(t.nbytes));
            if (!bridgespec_artifact::read_exact_at(wf, t.offset, hostbuf.data(), t.nbytes)) { fclose(wf); fprintf(stderr, "[hip-sidecar] short read %s\n", t.name.c_str()); return -1; }
            SC_CHECK(hipMalloc(&t.dev, static_cast<size_t>(t.nbytes)));
            SC_CHECK(hipMemcpy(t.dev, hostbuf.data(), static_cast<size_t>(t.nbytes), hipMemcpyHostToDevice));
        }
    }
    fclose(wf);
    SC_CHECK(hipMalloc(&S.remap_d, expected_remap_size));
    SC_CHECK(hipMemcpy(S.remap_d, remap_h.data(), expected_remap_size, hipMemcpyHostToDevice));

    auto & s = S;
    for (int i = 0; i < N_SCR; ++i) {
        auto & c = s.scr[i];
        SC_CHECK(hipMalloc(&c.x_tok, N_EMBD * 4));  SC_CHECK(hipMalloc(&c.x_h, N_EMBD * 4));
        SC_CHECK(hipMalloc(&c.e_n, N_EMBD * 4));    SC_CHECK(hipMalloc(&c.h_n, N_EMBD * 4));
        SC_CHECK(hipMalloc(&c.cat, 2 * N_EMBD * 4)); SC_CHECK(hipMalloc(&c.proj, N_EMBD * 4));
        SC_CHECK(hipMalloc(&c.attin, N_EMBD * 4));
        SC_CHECK(hipMalloc(&c.qfull, 2 * N_HEAD * HEAD_DIM * 4));
        SC_CHECK(hipMalloc(&c.q, N_HEAD * HEAD_DIM * 4)); SC_CHECK(hipMalloc(&c.gate, N_HEAD * HEAD_DIM * 4));
        SC_CHECK(hipMalloc(&c.k, N_HEAD_KV * HEAD_DIM * 4)); SC_CHECK(hipMalloc(&c.v, N_HEAD_KV * HEAD_DIM * 4));
        SC_CHECK(hipMalloc(&c.attout, N_HEAD * HEAD_DIM * 4));
        SC_CHECK(hipMalloc(&c.ffn_in, N_EMBD * 4)); SC_CHECK(hipMalloc(&c.ffn_g, N_FF * 4)); SC_CHECK(hipMalloc(&c.ffn_u, N_FF * 4));
        SC_CHECK(hipMalloc(&c.ffn_tmp, N_EMBD * 4));
        SC_CHECK(hipMalloc(&c.resid, N_EMBD * 4));  SC_CHECK(hipMalloc(&c.hnorm2, N_EMBD * 4));
        SC_CHECK(hipMalloc(&c.id_dev, 4));
        SC_CHECK(hipStreamCreate(&s.rst[i]));
    }
    {
        auto & b = s.bs;
        SC_CHECK(hipMalloc(&b.x_tok, N_BATCH * N_EMBD * 4));
        SC_CHECK(hipMalloc(&b.e_n,   N_BATCH * N_EMBD * 4));
        SC_CHECK(hipMalloc(&b.h_n,   N_BATCH * N_EMBD * 4));
        SC_CHECK(hipMalloc(&b.cat,   N_BATCH * 2 * N_EMBD * 4));
        SC_CHECK(hipMalloc(&b.proj,  N_BATCH * N_EMBD * 4));
        SC_CHECK(hipMalloc(&b.attin, N_BATCH * N_EMBD * 4));
        SC_CHECK(hipMalloc(&b.qfull, N_BATCH * 2 * N_HEAD * HEAD_DIM * 4));
        SC_CHECK(hipMalloc(&b.q,     N_BATCH * N_HEAD * HEAD_DIM * 4));
        SC_CHECK(hipMalloc(&b.gate,  N_BATCH * N_HEAD * HEAD_DIM * 4));
        SC_CHECK(hipMalloc(&b.k,     N_BATCH * N_HEAD_KV * HEAD_DIM * 4));
        SC_CHECK(hipMalloc(&b.v,     N_BATCH * N_HEAD_KV * HEAD_DIM * 4));
        SC_CHECK(hipMalloc(&b.attout, N_BATCH * N_HEAD * HEAD_DIM * 4));
        SC_CHECK(hipMalloc(&b.ffn_in, N_BATCH * N_EMBD * 4));
        SC_CHECK(hipMalloc(&b.ffn_g, N_BATCH * N_FF * 4));
        SC_CHECK(hipMalloc(&b.ffn_u, N_BATCH * N_FF * 4));
        SC_CHECK(hipMalloc(&b.ffn_tmp, N_BATCH * N_EMBD * 4));
        SC_CHECK(hipMalloc(&b.resid, N_BATCH * N_EMBD * 4));
        SC_CHECK(hipMalloc(&b.toks, N_BATCH * 4));
        SC_CHECK(hipMalloc(&b.poss, N_BATCH * 4));
    }
    SC_CHECK(hipMalloc(&s.part_m,   N_HEAD * ATTN_MAX_PARTS * 4));
    SC_CHECK(hipMalloc(&s.part_l,   N_HEAD * ATTN_MAX_PARTS * 4));
    SC_CHECK(hipMalloc(&s.part_acc, (size_t) N_HEAD * ATTN_MAX_PARTS * HEAD_DIM * 4));
    SC_CHECK(hipMalloc(&s.logits, N_HEAD_ROWS * 4));
    s.nparts = (N_HEAD_ROWS + 256 * 8 - 1) / (256 * 8);
    SC_CHECK(hipMalloc(&s.pmax, s.nparts * 4)); SC_CHECK(hipMalloc(&s.pidx, s.nparts * 4));
    SC_CHECK(hipMalloc(&s.tok_out, 8 * 4));
    if (const char * mp = getenv("LLAMA_SPEC_HIP_MAX_POS")) {
        const long long v = atoll(mp);
        // hard capacity: the split-attention partial buffers cover at most
        // ATTN_MAX_PARTS * ATTN_SPLIT positions; a larger max_pos would send the
        // kernel grid out of bounds. Clamp to [4096, that capacity] and cap at
        // the model's trained context.
        const long long cap = (long long) ATTN_MAX_PARTS * ATTN_SPLIT;
        long long c2 = v;
        if (c2 < 4096) c2 = 4096;
        if (c2 > cap) c2 = cap;
        if (c2 > 131072) c2 = 131072;
        if (c2 != v) fprintf(stderr, "[hip-sidecar] LLAMA_SPEC_HIP_MAX_POS=%lld clamped to %lld\n", v, c2);
        s.max_pos = (size_t) c2;
    }
    SC_CHECK(hipMalloc(&s.Kc, s.max_pos * N_HEAD_KV * HEAD_DIM * 2));
    SC_CHECK(hipMalloc(&s.Vc, s.max_pos * N_HEAD_KV * HEAD_DIM * 2));
    SC_CHECK(hipStreamCreate(&s.st));
    SC_CHECK(hipEventCreate(&s.ev_a));
    SC_CHECK(hipEventCreate(&s.ev_b));

    auto W  = [&](const char * n) { return (const block_q4_0 *) find_tensor(n)->dev; };
    auto Wf = [&](const char * n) { return (const float *) find_tensor(n)->dev; };
    s.W_embd = W("token_embd.weight");
    s.W_ehp  = W("blk.64.nextn.eh_proj.weight");
    s.w_enorm = Wf("blk.64.nextn.enorm.weight");
    s.w_hnorm = Wf("blk.64.nextn.hnorm.weight");
    s.w_attn  = Wf("blk.64.attn_norm.weight");
    s.W_q = W("blk.64.attn_q.weight");
    s.W_k = W("blk.64.attn_k.weight");
    s.W_v = W("blk.64.attn_v.weight");
    s.W_o = W("blk.64.attn_output.weight");
    s.w_qn = Wf("blk.64.attn_q_norm.weight");
    s.w_kn = Wf("blk.64.attn_k_norm.weight");
    s.w_post = Wf("blk.64.post_attention_norm.weight");
    s.W_fg = W("blk.64.ffn_gate.weight");
    s.W_fu = W("blk.64.ffn_up.weight");
    s.W_fd = W("blk.64.ffn_down.weight");
    s.w_shn = Wf("blk.64.nextn.shared_head_norm.weight");
    s.W_head = W("blk.64.nextn.shared_head_head.weight");

    S.ok = true;
    fprintf(stderr, "[hip-sidecar] initialized: %zu tensors, KV %zu positions f16\n", S.tensors.size(), S.max_pos);
    return 0;
}

// catch-up rows: token+pos+h per row; only KV side effects
SC_API int spec_hip_catchup(const int32_t * toks, const int32_t * poss, const float * h_rows, int n) {
    if (!S.ok || n < 0 || toks == nullptr || poss == nullptr || h_rows == nullptr) return -1;
    for (int r = 0; r < n; ++r) {
        if (toks[r] < 0 || toks[r] >= SC_N_VOCAB) {
            fprintf(stderr, "[hip-sidecar] catchup: invalid token %d at row %d\n", toks[r], r);
            return -1;
        }
    }
    static int n_calls = 0, n_rows = 0;
    static double total_ms = 0.0, total_gpu = 0.0;
    const bool dbg = getenv("LLAMA_SPEC_HIP_DEBUG") != nullptr;
    auto t0 = std::chrono::high_resolution_clock::now();
    if (dbg) SC_CHECK(hipEventRecord(S.ev_a, S.st));
    // all rows of a group run through each op in ONE batched launch; causality
    // is honored by writing every row's KV before the batched attention, whose
    // per-row length masks to that row's own position.
    for (int r = 0; r < n; ++r) {
        if (poss[r] < 0 || (size_t) poss[r] >= S.max_pos) return -1; // caller guarantees bounds
    }
    auto & b = S.bs;
    hipStream_t st = S.st;
    for (int r0 = 0; r0 < n; r0 += N_BATCH) {
        const int nb = std::min(N_BATCH, n - r0);
        SC_CHECK(hipMemcpyAsync(b.ffn_tmp, h_rows + (size_t) r0 * N_EMBD, (size_t) nb * N_EMBD * 4, hipMemcpyHostToDevice, st));
        SC_CHECK(hipMemcpyAsync(b.toks, toks + r0, nb * 4, hipMemcpyHostToDevice, st));
        SC_CHECK(hipMemcpyAsync(b.poss, poss + r0, nb * 4, hipMemcpyHostToDevice, st));
        // KV-ONLY (default): catch-up rows exist purely for their K/V cache side
        // effects. Nothing ever reads their attention output or FFN result (each
        // row's inputs come from the batch, not the previous row), so the Q path,
        // attention, output projection and the whole FFN are skipped — that
        // deletes ~194 MiB of weight reads per row.
        // LLAMA_SPEC_HIP_FULL_CATCHUP=1 restores the full block (isolation A/B).
        static const bool full_catchup = getenv("LLAMA_SPEC_HIP_FULL_CATCHUP") != nullptr;
        const dim3 gcol((N_EMBD + 255) / 256, nb);
        deq_rows_q4_0_b<<<gcol, 256, 0, st>>>(S.W_embd, b.toks, b.x_tok, N_EMBD);
        rmsnorm_b<<<nb, 256, 0, st>>>(b.x_tok, S.w_enorm, b.e_n, N_EMBD);
        rmsnorm_b<<<nb, 256, 0, st>>>(b.ffn_tmp, S.w_hnorm, b.h_n, N_EMBD);
        concat2_b<<<dim3((2 * N_EMBD + 255) / 256, nb), 256, 0, st>>>(b.e_n, b.h_n, b.cat, N_EMBD);
        gemv_q4_0_b<<<dim3(N_EMBD, nb), 32, 0, st>>>(S.W_ehp, b.cat, b.proj, 2 * N_EMBD, N_EMBD);
        rmsnorm_b<<<nb, 256, 0, st>>>(b.proj, S.w_attn, b.attin, N_EMBD);
        gemv_q4_0_b<<<dim3(N_HEAD_KV * HEAD_DIM, nb), 32, 0, st>>>(S.W_k, b.attin, b.k, N_EMBD, N_HEAD_KV * HEAD_DIM);
        gemv_q4_0_b<<<dim3(N_HEAD_KV * HEAD_DIM, nb), 32, 0, st>>>(S.W_v, b.attin, b.v, N_EMBD, N_HEAD_KV * HEAD_DIM);
        rmsnorm_heads_b<<<dim3(N_HEAD_KV, nb), 256, 0, st>>>(b.k, S.w_kn, b.k, N_HEAD_KV);
        rope_neox_b<<<dim3(N_HEAD_KV, nb), N_ROT / 2, 0, st>>>(b.k, N_HEAD_KV, b.poss);
        kv_store_b<<<dim3((N_HEAD_KV * HEAD_DIM + 255) / 256, nb), 256, 0, st>>>(b.k, b.v, S.Kc, S.Vc, b.poss);
        if (full_catchup) {
            gemv_q4_0_b<<<dim3(2 * N_HEAD * HEAD_DIM, nb), 32, 0, st>>>(S.W_q, b.attin, b.qfull, N_EMBD, 2 * N_HEAD * HEAD_DIM);
            split_q_gate_b<<<dim3(N_HEAD, nb), HEAD_DIM, 0, st>>>(b.qfull, b.q, b.gate);
            rmsnorm_heads_b<<<dim3(N_HEAD, nb), 256, 0, st>>>(b.q, S.w_qn, b.q, N_HEAD);
            rope_neox_b<<<dim3(N_HEAD, nb), N_ROT / 2, 0, st>>>(b.q, N_HEAD, b.poss);
            attn_1q_b<<<dim3(N_HEAD, nb), 256, 0, st>>>(b.q, S.Kc, S.Vc, b.attout, b.poss);
            sigmoid_gate<<<(nb * N_HEAD * HEAD_DIM + 255) / 256, 256, 0, st>>>(b.attout, b.gate, nb * N_HEAD * HEAD_DIM);
            gemv_q4_0_b<<<dim3(N_EMBD, nb), 32, 0, st>>>(S.W_o, b.attout, b.resid, N_HEAD * HEAD_DIM, N_EMBD);
            add_inplace<<<(nb * N_EMBD + 255) / 256, 256, 0, st>>>(b.resid, b.proj, nb * N_EMBD);
            rmsnorm_b<<<nb, 256, 0, st>>>(b.resid, S.w_post, b.ffn_in, N_EMBD);
            gemv_q4_0_b<<<dim3(N_FF, nb), 32, 0, st>>>(S.W_fg, b.ffn_in, b.ffn_g, N_EMBD, N_FF);
            gemv_q4_0_b<<<dim3(N_FF, nb), 32, 0, st>>>(S.W_fu, b.ffn_in, b.ffn_u, N_EMBD, N_FF);
            silu_mul<<<(nb * N_FF + 255) / 256, 256, 0, st>>>(b.ffn_g, b.ffn_u, nb * N_FF);
            gemv_q4_0_b<<<dim3(N_EMBD, nb), 32, 0, st>>>(S.W_fd, b.ffn_g, b.ffn_tmp, N_FF, N_EMBD);
        }
        SC_CHECK(hipStreamSynchronize(st));
    }
    if (dbg) {
        SC_CHECK(hipEventRecord(S.ev_b, S.st));
        SC_CHECK(hipEventSynchronize(S.ev_b));
        float gpu = 0.0f;
        SC_CHECK(hipEventElapsedTime(&gpu, S.ev_a, S.ev_b));
        total_gpu += gpu;
        auto t1 = std::chrono::high_resolution_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        n_rows += n;
        if (++n_calls % 25 == 0) {
            fprintf(stderr, "[hip-sidecar] catchup avg %.2f ms wall / %.2f ms gpu (%.1f rows/call) over %d calls\n",
                    total_ms / n_calls, total_gpu / n_calls, (double) n_rows / n_calls, n_calls);
        }
    }
    return 0;
}

// draft n tokens greedily from (id_last, n_past, h); out_ids receives full-vocab ids
SC_API int spec_hip_draft(int32_t id_last, int32_t n_past, const float * h, int n, int32_t * out_ids) {
    if (!S.ok || n < 1 || n > 8 || h == nullptr || out_ids == nullptr) return -1;
    if (id_last < 0 || id_last >= SC_N_VOCAB) {
        fprintf(stderr, "[hip-sidecar] draft: invalid id_last %d\n", id_last);
        return -1;
    }
    if (n_past < 0 || static_cast<size_t>(n_past) + static_cast<size_t>(n) > S.max_pos) return -1;
    // fault-injection hook for hardening tests: returns garbage ids once
    if (getenv("LLAMA_SPEC_HIP_FAULT_BADID") != nullptr) {
        static bool fired = false;
        if (!fired) {
            fired = true;
            for (int j = 0; j < n; ++j) out_ids[j] = 999999999;
            return 0;
        }
    }
    static int n_calls = 0;
    static double total_ms = 0.0, total_gpu = 0.0;
    const bool dbg = getenv("LLAMA_SPEC_HIP_DEBUG") != nullptr;
    auto t0 = std::chrono::high_resolution_clock::now();
    if (dbg) SC_CHECK(hipEventRecord(S.ev_a, S.st));
    auto & c = S.scr[0];
    SC_CHECK(hipMemcpyAsync(c.x_h, h, N_EMBD * 4, hipMemcpyHostToDevice, S.st));
    SC_CHECK(hipMemcpyAsync(c.id_dev, &id_last, 4, hipMemcpyHostToDevice, S.st));
    for (int j = 0; j < n; ++j) {
        const int * tok_src = j == 0 ? c.id_dev : S.tok_out + (j - 1);
        const float * xh_in = j == 0 ? c.x_h : c.hnorm2;
        deq_row_q4_0<<<(N_EMBD + 255) / 256, 256, 0, S.st>>>(S.W_embd, tok_src, c.x_tok, N_EMBD, SC_N_VOCAB);
        block_phase1(c, S.st, xh_in, n_past + j);
        block_phase2(c, S.st, n_past + j, S.tok_out + j);
    }
    if (dbg) SC_CHECK(hipEventRecord(S.ev_b, S.st));
    SC_CHECK(hipMemcpyAsync(out_ids, S.tok_out, 4 * n, hipMemcpyDeviceToHost, S.st));
    SC_CHECK(hipStreamSynchronize(S.st));
    for (int j = 0; j < n; ++j) {
        if (out_ids[j] < 0 || out_ids[j] >= SC_N_VOCAB) {
            fprintf(stderr, "[hip-sidecar] draft produced invalid token at step %d\n", j);
            return -1;
        }
    }
    if (dbg) {
        float gpu = 0.0f;
        SC_CHECK(hipEventElapsedTime(&gpu, S.ev_a, S.ev_b));
        // rolling 25-call window, not lifetime-cumulative: deep-position costs must
        // not be diluted by earlier shallow calls (audit correction)
        static double win_gpu = 0.0, win_wall = 0.0;
        static int win_n = 0;
        auto t1 = std::chrono::high_resolution_clock::now();
        win_gpu  += gpu;
        win_wall += std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_gpu += gpu;
        total_ms  += std::chrono::duration<double, std::milli>(t1 - t0).count();
        n_calls++;
        if (++win_n == 25) {
            fprintf(stderr, "[hip-sidecar] draft last25 %.2f ms wall / %.2f ms gpu @pos=%d (lifetime %.2f/%.2f over %d)\n",
                    win_wall / win_n, win_gpu / win_n, n_past, total_ms / n_calls, total_gpu / n_calls, n_calls);
            win_gpu = win_wall = 0.0;
            win_n = 0;
        }
    }
    return 0;
}

// ---------------- harness: replay golden.bin through the API ----------------
#ifdef SIDECAR_HARNESS
int main(int argc, char ** argv) {
    const char * wdir = argc > 1 ? argv[1] : ".";
    const char * ids_path = argc > 2 ? argv[2] : "draft_head_ids.bin";
    const char * golden_path = argc > 3 ? argv[3] : nullptr;
    if (spec_hip_init(wdir, ids_path) != 0) return 1;

    char pbuf[1024];
    if (golden_path) {
        snprintf(pbuf, sizeof(pbuf), "%s", golden_path);
    } else {
        snprintf(pbuf, sizeof(pbuf), "%s/golden.bin", wdir);
    }
    FILE * g = fopen(pbuf, "rb");
    if (!g) { printf("cannot open golden.bin\n"); return 1; }
    uint32_t magic, ne;
    fread(&magic, 4, 1, g); fread(&ne, 4, 1, g);
    if (magic != 0x50444853 || ne != (uint32_t) N_EMBD) { printf("bad golden header\n"); return 1; }

    std::vector<float> hrow(N_EMBD);
    int cycles = 0, tok_match = 0, tok_total = 0, cycles_exact = 0;
    int pend_id_last = 0, pend_n_past = 0;
    bool have_draft_in = false;
    std::vector<int32_t> my_ids;
    double gpu_ms = 0.0;

    while (true) {
        uint32_t typ;
        if (fread(&typ, 4, 1, g) != 1) break;
        if (typ == 0) {
            uint32_t n; fread(&n, 4, 1, g);
            std::vector<int32_t> toks(n), poss(n);
            std::vector<float> hs((size_t) n * N_EMBD);
            for (uint32_t r = 0; r < n; ++r) {
                fread(&toks[r], 4, 1, g); fread(&poss[r], 4, 1, g);
                fread(hs.data() + (size_t) r * N_EMBD, 4, N_EMBD, g);
            }
            spec_hip_catchup(toks.data(), poss.data(), hs.data(), (int) n);
        } else if (typ == 1) {
            fread(&pend_id_last, 4, 1, g); fread(&pend_n_past, 4, 1, g);
            fread(hrow.data(), 4, N_EMBD, g);
            have_draft_in = true;
            my_ids.assign(3, -1);
            auto t0 = std::chrono::high_resolution_clock::now();
            spec_hip_draft(pend_id_last, pend_n_past, hrow.data(), 3, my_ids.data());
            auto t1 = std::chrono::high_resolution_clock::now();
            gpu_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        } else if (typ == 2) {
            uint32_t n; fread(&n, 4, 1, g);
            std::vector<int32_t> exp_ids(n);
            if (n) fread(exp_ids.data(), 4, n, g);
            if (!have_draft_in) continue;
            have_draft_in = false;
            cycles++;
            bool exact = (n == my_ids.size());
            for (uint32_t i = 0; i < n; ++i) {
                tok_total++;
                if (i < my_ids.size() && my_ids[i] == exp_ids[i]) tok_match++;
                else exact = false;
            }
            if (exact) cycles_exact++;
        }
    }
    fclose(g);
    printf("PARITY(api): %d/%d cycles exact (%.1f%%), %d/%d tokens (%.1f%%)\n",
           cycles_exact, cycles, 100.0 * cycles_exact / (cycles ? cycles : 1),
           tok_match, tok_total, 100.0 * tok_match / (tok_total ? tok_total : 1));
    printf("draft wall: %.2f ms per draft\n", gpu_ms / (cycles ? cycles : 1));
    return 0;
}
#endif
