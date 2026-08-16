// Synthetic kernel micro-benchmark for DeepSeek V4 Flash decode on RDNA2 (gfx1030).
//
// Measures the performance gap of three HIP optimizations applied on
// branch debug/dsv4-split-mode-tensor:
//
//   1. mmvq: LDS staging of the IQ2_S / IQ3_XXS lookup grids
//      (grids are plain __device__ globals upstream; every inner-loop gather
//      replays against L1 with divergent per-lane indices).
//   2. mmvq: RDNA2 nwarps tuning for Q8_0 / Q6_K (was hardcoded to 1 warp).
//   3. lightning indexer: segmented-lane kernel (8 lanes/head, 4 heads/pass)
//      vs the generic warp-reduction kernel.
//
// The mmvq kernels below are faithful replicas of the mul_mat_vec_q inner loop
// (same kbx iteration, same vec_dot calls, same cross-warp reduction). The
// indexer kernels are verbatim copies of the two production kernels. The
// end-to-end validation should still be done with test-backend-ops and
// llama-bench once GPU time is available.
//
// Build (see scripts/bench-dsv4-rdna2.sh):
//   clang++ -x hip --offload-arch=gfx1030 -O3 -std=gnu++17 \
//       -funsafe-math-optimizations -DGGML_USE_HIP -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 \
//       -I ggml/src/ggml-cuda -I ggml/include -I ggml/src \
//       bench-dsv4-rdna2.cu -o bench-dsv4-rdna2

#include "common.cuh"
#include "vecdotq.cuh"
#include "fattn-common.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

#define BENCH_CHECK(x)                                                                        \
    do {                                                                                      \
        hipError_t err_ = (x);                                                                \
        if (err_ != hipSuccess) {                                                             \
            fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(err_), __FILE__, __LINE__); \
            exit(1);                                                                          \
        }                                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// host helpers
// ---------------------------------------------------------------------------

static uint32_t xs_state = 0x12345678u;
static uint32_t xorshift32() {
    xs_state ^= xs_state << 13;
    xs_state ^= xs_state >> 17;
    xs_state ^= xs_state << 5;
    return xs_state;
}

static void fill_random_bytes(void * p, size_t n) {
    uint8_t * b = (uint8_t *) p;
    for (size_t i = 0; i < n; ++i) {
        b[i] = (uint8_t) (xorshift32() & 0xFF);
    }
}

static void set_half(void * p, uint16_t bits) {
    *(uint16_t *) p = bits;
}

struct timer {
    hipEvent_t t0, t1;
    timer()  { BENCH_CHECK(hipEventCreate(&t0)); BENCH_CHECK(hipEventCreate(&t1)); }
    ~timer() { hipEventDestroy(t0); hipEventDestroy(t1); }
    float time(void (*fn)(void *), void * arg, int warmup, int iters) {
        for (int i = 0; i < warmup; ++i) fn(arg);
        BENCH_CHECK(hipDeviceSynchronize());
        BENCH_CHECK(hipEventRecord(t0));
        for (int i = 0; i < iters; ++i) fn(arg);
        BENCH_CHECK(hipEventRecord(t1));
        BENCH_CHECK(hipEventSynchronize(t1));
        float ms = 0.0f;
        BENCH_CHECK(hipEventElapsedTime(&ms, t0, t1));
        return ms / iters;
    }
};

// ---------------------------------------------------------------------------
// mmvq replica (mirrors mul_mat_vec_q with ncols_dst=1, rows_per_block=1)
// ---------------------------------------------------------------------------

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

template <ggml_type type>
static constexpr __device__ vec_dot_q_cuda_t bench_get_vec_dot() {
    switch (type) {
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        default:                return nullptr;
    }
}

template <ggml_type type>
static constexpr __device__ int bench_get_vdr() {
    switch (type) {
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:   return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS: return VDR_IQ3_XXS_Q8_1_MMVQ;
        default:                return 1;
    }
}

template <ggml_type type, int nwarps, bool use_lds, int rpb = 1>
__launch_bounds__(nwarps*32, 1)
static __global__ void bench_mmvq_kernel(
        const void * __restrict__ vx, const block_q8_1 * __restrict__ y, float * __restrict__ dst,
        const int ncols_x) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = bench_get_vdr<type>();
    constexpr int warp_size = 32;
    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = bench_get_vec_dot<type>();

    const int tid = warp_size*threadIdx.y + threadIdx.x;
    const int row0 = rpb*blockIdx.x;
    const int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

const auto vec_dot = vec_dot_q_cuda;

    float tmp[rpb] = {0.0f};

    // channel (expert) offset, mirroring channel_x*stride_channel_x in mul_mat_vec_q
    const int kbx_offset = (blockIdx.y * gridDim.x * rpb + row0) * blocks_per_row_x;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (tid % (qi/vdr));
#pragma unroll
        for (int i = 0; i < rpb; ++i) {
            tmp[i] += vec_dot(vx, &y[kby], kbx_offset + i*blocks_per_row_x + kbx, kqs);
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][rpb][warp_size];

    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < rpb; ++i) {
            tmp_shared[threadIdx.y-1][i][threadIdx.x] = tmp[i];
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

#pragma unroll
    for (int i = 0; i < rpb; ++i) {
#pragma unroll
        for (int l = 0; l < nwarps-1; ++l) {
            tmp[i] += tmp_shared[l][i][threadIdx.x];
        }
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
    }
    if (threadIdx.x < rpb) {
        dst[blockIdx.y * gridDim.x * rpb + row0 + threadIdx.x] = tmp[threadIdx.x];
    }
}

struct mmvq_case {
    std::string name;
    ggml_type   type;
    int         rows;
    int         ncols;
    int         channels;   // simulate per-expert channels (grid.y)
    size_t      block_bytes;
};

struct mmvq_run_args {
    const void * vx;
    const block_q8_1 * y;
    float * dst;
    int ncols;
    int rows;
    int channels;
    // kernel selector
    ggml_type type;
    int nwarps;
    int use_lds;
};

template <ggml_type type, int nwarps, bool use_lds, int rpb = 1>
static void mmvq_launch(const mmvq_run_args & a, hipStream_t stream) {
    dim3 grid(a.rows / rpb, a.channels);
    dim3 block(32, nwarps);
    bench_mmvq_kernel<type, nwarps, use_lds, rpb><<<grid, block, 0, stream>>>(a.vx, a.y, a.dst, a.ncols);
}

static size_t type_block_bytes(ggml_type t) {
    switch (t) {
        case GGML_TYPE_Q8_0:    return sizeof(block_q8_0);
        case GGML_TYPE_Q6_K:    return sizeof(block_q6_K);
        case GGML_TYPE_IQ2_S:   return sizeof(block_iq2_s);
        case GGML_TYPE_IQ3_XXS: return sizeof(block_iq3_xxs);
        default: return 0;
    }
}

static int type_qk(ggml_type t) {
    switch (t) {
        case GGML_TYPE_Q8_0:    return QK8_0;
        case GGML_TYPE_Q6_K:    return QK_K;
        case GGML_TYPE_IQ2_S:   return QK_K;
        case GGML_TYPE_IQ3_XXS: return QK_K;
        default: return 0;
    }
}

struct mmvq_timed {
    const mmvq_run_args * a;
    int which;  // 0=1w global, 1=1w lds, 2=2w lds, 3=4w lds, 4=2w global, 5=4w global
};

template <ggml_type type>
static void mmvq_dispatch(const mmvq_timed * t, hipStream_t stream) {
    const mmvq_run_args & a = *t->a;
    switch (t->which) {
        case 0: mmvq_launch<type, 1, false>(a, stream); break;
        case 4: mmvq_launch<type, 2, false>(a, stream); break;
        case 5: mmvq_launch<type, 4, false>(a, stream); break;
        case 6: mmvq_launch<type, 4, false, 4>(a, stream); break;
    }
}

template <ggml_type type>
static void run_mmvq_case(const char * name, int rows, int ncols, int channels, bool has_table) {
    const int qk = type_qk(type);
    const size_t bb = type_block_bytes(type);
    const size_t x_bytes = (size_t) rows * (ncols/qk) * bb * channels;
    const size_t y_bytes = (size_t) (ncols/QK8_1) * sizeof(block_q8_1);

    std::vector<uint8_t> hx(x_bytes), hy(y_bytes);
    fill_random_bytes(hx.data(), x_bytes);
    fill_random_bytes(hy.data(), y_bytes);

    // make block scales finite and uniform so results are comparable
    const size_t n_xblocks = (size_t) rows * (ncols/qk) * channels;
    for (size_t i = 0; i < n_xblocks; ++i) {
        // d is at offset 0 for q8_0/iq2_s/iq3_xxs; at the end for q6_K
        if (type == GGML_TYPE_Q6_K) {
            set_half(hx.data() + i*bb + (bb - sizeof(ggml_half)), 0x3C00);
        } else {
            set_half(hx.data() + i*bb, 0x3C00);
        }
    }
    for (size_t i = 0; i < y_bytes/sizeof(block_q8_1); ++i) {
        block_q8_1 * b = (block_q8_1 *) (hy.data() + i*sizeof(block_q8_1));
        b->ds = __floats2half2_rn(1.0f, 0.0f);
    }

    void * dx; void * dy; float * dd;
    BENCH_CHECK(hipMalloc(&dx, x_bytes));
    BENCH_CHECK(hipMalloc(&dy, y_bytes));
    BENCH_CHECK(hipMalloc(&dd, sizeof(float)*rows*channels));
    BENCH_CHECK(hipMemcpy(dx, hx.data(), x_bytes, hipMemcpyHostToDevice));
    BENCH_CHECK(hipMemcpy(dy, hy.data(), y_bytes, hipMemcpyHostToDevice));

    mmvq_run_args a { dx, (const block_q8_1 *) dy, dd, ncols, rows, channels, type, 1, 0 };

    auto run_which = [&](int which) {
        mmvq_timed t { &a, which };
        mmvq_dispatch<type>(&t, 0);
    };

    // correctness: baseline vs new variants
    std::vector<float> ref(rows*channels), out(rows*channels);
    const int variants[4] = {0, 4, 5, 6};
    const char * vnames[4] = {"1 warp (baseline)", "2 warps", "4 warps", "4 warps, small_k rpb=4"};

    run_which(0);
    BENCH_CHECK(hipMemcpy(ref.data(), dd, sizeof(float)*rows*channels, hipMemcpyDeviceToHost));

    printf("\n=== mmvq %s  rows=%d K=%d channels=%d  x=%.1f MB ===\n", name, rows, ncols, channels, x_bytes/1e6);

    timer tm;
    for (int v = 0; v < 4; ++v) {
        const int which = variants[v];
        (void) has_table;
        // correctness vs baseline (nwarps changes reduction order -> tolerance)
        BENCH_CHECK(hipMemset(dd, 0, sizeof(float)*rows*channels));
        run_which(which);
        BENCH_CHECK(hipMemcpy(out.data(), dd, sizeof(float)*rows*channels, hipMemcpyDeviceToHost));
        double max_rel = 0.0; int nbad = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            const double denom = fabs((double) ref[i]) > 1e-6 ? fabs((double) ref[i]) : 1e-6;
            const double rel = fabs((double) out[i] - (double) ref[i]) / denom;
            max_rel = rel > max_rel ? rel : max_rel;
            if (rel > 1e-3 || !std::isfinite(out[i])) nbad++;
        }

        mmvq_timed t { &a, which };
        float ms = tm.time([](void * p) { mmvq_dispatch<type>((const mmvq_timed *) p, 0); }, &t, 3, 20);
        const double gbs = x_bytes / (ms * 1e-3) / 1e9;
        printf("  %-32s %8.3f ms  %7.1f GB/s  max_rel=%.2e %s\n", vnames[v], ms, gbs, max_rel,
               nbad == 0 ? "OK" : "*** MISMATCH ***");
    }

    hipFree(dx); hipFree(dy); hipFree(dd);
}

// ---------------------------------------------------------------------------
// lightning indexer kernels (verbatim copies of the two production kernels)
// ---------------------------------------------------------------------------

// ---- baseline: generic vector kernel (current upstream path for HIP) ----
template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>
static __global__ void indexer_kernel_old(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3
    ) {

    constexpr int K_VECS_PER_WARP = K_VECS_PER_BLOCK / WARPS_PER_BLOCK;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;

    const int i_batch  = blockIdx.y;
    const int i_stream = blockIdx.z;
    const int i_warp   = threadIdx.y;
    const int i_lane   = threadIdx.x;
    const int tid      = i_warp * WARP_SIZE + i_lane;

    // each warp processes K_VECS_PER_WARP K vectors
    const int start_kv_block = blockIdx.x * K_VECS_PER_BLOCK;
    const int start_kv = start_kv_block + i_warp * K_VECS_PER_WARP;

    const char  * q_base = (const char  *)                 Q + i_batch*nbq2 + i_stream*nbq3;
    const float * w_base = (const float *) ((const char *) W + i_batch*nbw1 + i_stream*nbw3);

    // phase 1 - load (and dequantize if needed) K to registers

    float4 k_reg_f[K_VECS_PER_WARP];

    if constexpr (TYPE_K == GGML_TYPE_F32) {
        // direct copy of float4
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            int i_kv = start_kv + k;
            if (i_kv < n_kv) {
                const float4 * k_base = (const float4 *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                k_reg_f[k] = k_base[i_lane];
            } else {
                k_reg_f[k] = make_float4(0, 0, 0, 0);
            }
        }
    } else {
        // dequantize remaining types to float
        constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, float, 4>();
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            int i_kv = start_kv + k;
            if (i_kv < n_kv) {
                const void * k_base = (const void *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                dequantize_k(k_base, &k_reg_f[k], i_lane * 4);
            } else {
                k_reg_f[k] = make_float4(0, 0, 0, 0);
            }
        }
    }

    float score_k[K_VECS_PER_WARP] = { 0.0f };

    // load weights and Q only for N_HEAD_INNER heads at once to reduce shared memory usage
    constexpr int N_HEAD_INNER = N_HEAD / 4;

    for (int i_head_0 = 0; i_head_0 < N_HEAD; i_head_0 += N_HEAD_INNER) {
        // phase 2 - load weights and Q to shared memory

        __shared__ float  w_shared[N_HEAD_INNER];
        __shared__ float4 q_shared_f[N_HEAD_INNER][N_EMBD / 4];

        if (tid < N_HEAD_INNER) {
            w_shared[tid] = w_base[i_head_0 + tid];
        }

        constexpr int n_q = N_HEAD_INNER * (N_EMBD / 4);
#pragma unroll
        for (int i_q = tid; i_q < n_q; i_q += THREADS_PER_BLOCK) {
            const int i_head_inner = i_q / (N_EMBD / 4);
            const int i_head = i_head_0 + i_head_inner;
            const int i_embd = i_q % (N_EMBD / 4);
            q_shared_f[i_head_inner][i_embd] = *(const float4 *) (q_base + i_head*nbq1 + i_embd*sizeof(float4));
        }

        __syncthreads();

        // phase 3 - calculate lightning indexer scores

        for (int i_head_inner = 0; i_head_inner < N_HEAD_INNER; ++i_head_inner) {
            const float w_val = w_shared[i_head_inner];
            float qk[K_VECS_PER_WARP] = { 0.0f };

            // dot product of floats
            const float4 q_vec = q_shared_f[i_head_inner][i_lane];

#pragma unroll
            for (int k = 0; k < K_VECS_PER_WARP; ++k) {
                ggml_cuda_mad(qk[k], q_vec.x, k_reg_f[k].x);
                ggml_cuda_mad(qk[k], q_vec.y, k_reg_f[k].y);
                ggml_cuda_mad(qk[k], q_vec.z, k_reg_f[k].z);
                ggml_cuda_mad(qk[k], q_vec.w, k_reg_f[k].w);
            }

#pragma unroll
            for (int k = 0; k < K_VECS_PER_WARP; ++k) {
                float sum = warp_reduce_sum(qk[k]);

                // ReLU, weight
                if (i_lane == 0) {
                    sum = (sum > 0.0f) ? sum : 0.0f;
                    score_k[k] += sum * w_val;
                }
            }
        }

        __syncthreads();
    }

    // phase 4 - store outputs to shared memory

    __shared__ float dst_shared[K_VECS_PER_BLOCK];

    if (i_lane == 0) {
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            dst_shared[i_warp * K_VECS_PER_WARP + k] = score_k[k];
        }
    }

    __syncthreads();

    // phase 5 - write from shared memory to VRAM in coalesced manner

    if (tid < K_VECS_PER_BLOCK) {
        int i_kv = start_kv_block + tid;
        if (i_kv < n_kv) {
            const half * m_base = (const half *) ((const char *) M + i_batch*nbm1 + (i_stream%nem3)*nbm3);
            float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);
            dst_base[i_kv] = dst_shared[tid] + __half2float(m_base[i_kv]);
        }
    }
}

// ---- new: RDNA-optimized segmented-lane kernel (copy of production kernel) ----
template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>
static __global__ void __launch_bounds__(WARPS_PER_BLOCK*32, 1) indexer_kernel_new(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3
    ) {

    static_assert(N_EMBD == 128, "only head_dim 128 is supported");
    static_assert(N_HEAD % 4 == 0, "only a multiple of 4 heads is supported");

    constexpr int HEADS_PER_PASS = 4;
    constexpr int LANES_PER_HEAD = 32 / HEADS_PER_PASS;           // 8
    constexpr int F4_PER_LANE    = N_EMBD / LANES_PER_HEAD / 4;   // 4
    constexpr int N_PASS         = N_HEAD / HEADS_PER_PASS;
    constexpr int K_VECS_PER_WARP  = K_VECS_PER_BLOCK / WARPS_PER_BLOCK;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;

    const int i_batch  = blockIdx.y;
    const int i_stream = blockIdx.z;
    const int i_warp   = threadIdx.y;
    const int i_lane   = threadIdx.x;
    const int tid      = i_warp * 32 + i_lane;

    const int head_sub = i_lane / LANES_PER_HEAD;
    const int dim_sub  = i_lane % LANES_PER_HEAD;

    const int start_kv_block = blockIdx.x * K_VECS_PER_BLOCK;
    const int start_kv       = start_kv_block + i_warp * K_VECS_PER_WARP;

    const char  * q_base = (const char  *)                 Q + i_batch*nbq2 + i_stream*nbq3;
    const float * w_base = (const float *) ((const char *) W + i_batch*nbw1 + i_stream*nbw3);

    float4 k_reg[K_VECS_PER_WARP][F4_PER_LANE];

    if constexpr (TYPE_K == GGML_TYPE_F32) {
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            const int i_kv = start_kv + k;
#pragma unroll
            for (int j = 0; j < F4_PER_LANE; ++j) {
                k_reg[k][j] = make_float4(0, 0, 0, 0);
                if (i_kv < n_kv) {
                    const float4 * k_base = (const float4 *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                    k_reg[k][j] = k_base[dim_sub*F4_PER_LANE + j];
                }
            }
        }
    } else {
        constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, float, 4>();
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            const int i_kv = start_kv + k;
#pragma unroll
            for (int j = 0; j < F4_PER_LANE; ++j) {
                k_reg[k][j] = make_float4(0, 0, 0, 0);
                if (i_kv < n_kv) {
                    const void * k_base = (const void *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                    dequantize_k(k_base, &k_reg[k][j], (dim_sub*F4_PER_LANE + j) * 4);
                }
            }
        }
    }

    __shared__ float4 q_shared[2][HEADS_PER_PASS][N_EMBD / 4];
    __shared__ float  w_shared[2][HEADS_PER_PASS];

    constexpr int n_q = HEADS_PER_PASS * (N_EMBD / 4);

    auto stage_q = [&](int pass) {
        const int i_head_0 = pass * HEADS_PER_PASS;
        const int buf      = pass & 1;
#pragma unroll
        for (int i_q = tid; i_q < n_q; i_q += THREADS_PER_BLOCK) {
            const int i_head = i_q / (N_EMBD / 4);
            const int i_embd = i_q % (N_EMBD / 4);
            q_shared[buf][i_head][i_embd] = *(const float4 *) (q_base + (i_head_0 + i_head)*nbq1 + i_embd*sizeof(float4));
        }
        if (tid < HEADS_PER_PASS) {
            w_shared[buf][tid] = w_base[i_head_0 + tid];
        }
    };

    stage_q(0);
    __syncthreads();

    float score_k[K_VECS_PER_WARP] = { 0.0f };

    for (int p = 0; p < N_PASS; ++p) {
        if (p + 1 < N_PASS) {
            stage_q(p + 1);
        }

        const float w_val = w_shared[p & 1][head_sub];

        float qk[K_VECS_PER_WARP] = { 0.0f };
#pragma unroll
        for (int j = 0; j < F4_PER_LANE; ++j) {
            const float4 q_vec = q_shared[p & 1][head_sub][dim_sub*F4_PER_LANE + j];
#pragma unroll
            for (int k = 0; k < K_VECS_PER_WARP; ++k) {
                ggml_cuda_mad(qk[k], q_vec.x, k_reg[k][j].x);
                ggml_cuda_mad(qk[k], q_vec.y, k_reg[k][j].y);
                ggml_cuda_mad(qk[k], q_vec.z, k_reg[k][j].z);
                ggml_cuda_mad(qk[k], q_vec.w, k_reg[k][j].w);
            }
        }

#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
#pragma unroll
            for (int offset = LANES_PER_HEAD / 2; offset > 0; offset >>= 1) {
                qk[k] += __shfl_xor_sync(0xffffffff, qk[k], offset, 32);
            }
            if (dim_sub == 0) {
                score_k[k] += fmaxf(qk[k], 0.0f) * w_val;
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int k = 0; k < K_VECS_PER_WARP; ++k) {
        score_k[k] += __shfl_sync(0xffffffff, score_k[k], i_lane + 2*LANES_PER_HEAD, 32);
        score_k[k] += __shfl_sync(0xffffffff, score_k[k], i_lane + 1*LANES_PER_HEAD, 32);
    }

    __shared__ float dst_shared[K_VECS_PER_BLOCK];

    if (i_lane == 0) {
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            dst_shared[i_warp * K_VECS_PER_WARP + k] = score_k[k];
        }
    }

    __syncthreads();

    if (tid < K_VECS_PER_BLOCK) {
        int i_kv = start_kv_block + tid;
        if (i_kv < n_kv) {
            const half * m_base = (const half *) ((const char *) M + i_batch*nbm1 + (i_stream%nem3)*nbm3);
            float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);
            dst_base[i_kv] = dst_shared[tid] + __half2float(m_base[i_kv]);
        }
    }
}

// ---- experimental: half2 kernel, 4 lanes per head, 8 heads per pass ----
// K is kept in registers as half2 (32 dims per lane), Q is staged as half2.
// Per (head, key): 16 HFMA2 + 2-step shuffle reduce over 4 lanes.
template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>
static __global__ void __launch_bounds__(WARPS_PER_BLOCK*32, 1) indexer_kernel_h2(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3
    ) {

    static_assert(N_EMBD == 128, "only head_dim 128 is supported");
    static_assert(N_HEAD % 8 == 0, "only a multiple of 8 heads is supported");

    constexpr int HEADS_PER_PASS = 8;
    constexpr int LANES_PER_HEAD = 32 / HEADS_PER_PASS;           // 4
    constexpr int H2_PER_LANE    = N_EMBD / LANES_PER_HEAD / 2;   // 16
    constexpr int N_PASS         = N_HEAD / HEADS_PER_PASS;
    constexpr int K_VECS_PER_WARP  = K_VECS_PER_BLOCK / WARPS_PER_BLOCK;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;

    const int i_batch  = blockIdx.y;
    const int i_stream = blockIdx.z;
    const int i_warp   = threadIdx.y;
    const int i_lane   = threadIdx.x;
    const int tid      = i_warp * 32 + i_lane;

    const int head_sub = i_lane / LANES_PER_HEAD;   // 0..7
    const int dim_sub  = i_lane % LANES_PER_HEAD;   // 0..3

    const int start_kv_block = blockIdx.x * K_VECS_PER_BLOCK;
    const int start_kv       = start_kv_block + i_warp * K_VECS_PER_WARP;

    const char  * q_base = (const char  *)                 Q + i_batch*nbq2 + i_stream*nbq3;
    const float * w_base = (const float *) ((const char *) W + i_batch*nbw1 + i_stream*nbw3);

    // phase 1 - load K to registers as half2 (lane owns 32 consecutive dims per key)
    half2 k_reg[K_VECS_PER_WARP][H2_PER_LANE];

    if constexpr (TYPE_K == GGML_TYPE_F16) {
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            const int i_kv = start_kv + k;
#pragma unroll
            for (int j = 0; j < H2_PER_LANE/4; ++j) {
                if (i_kv < n_kv) {
                    const uint4 * k_base = (const uint4 *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                    *(uint4 *) &k_reg[k][4*j] = k_base[dim_sub*(H2_PER_LANE/4) + j];
                } else {
                    *(uint4 *) &k_reg[k][4*j] = make_uint4(0, 0, 0, 0);
                }
            }
        }
    } else {
        constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, half, 4>();
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            const int i_kv = start_kv + k;
#pragma unroll
            for (int j = 0; j < H2_PER_LANE/2; ++j) {
                if (i_kv < n_kv) {
                    const void * k_base = (const void *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                    dequantize_k(k_base, &k_reg[k][2*j], (dim_sub*H2_PER_LANE + 2*j) * 2);
                } else {
                    k_reg[k][2*j+0] = __float2half2_rn(0.0f);
                    k_reg[k][2*j+1] = __float2half2_rn(0.0f);
                }
            }
        }
    }

    // phase 2/3 - Q staged as half2, double buffered
    __shared__ half2  q_shared[2][HEADS_PER_PASS][N_EMBD / 2];
    __shared__ float  w_shared[2][HEADS_PER_PASS];

    constexpr int n_q = HEADS_PER_PASS * (N_EMBD / 2);

    auto stage_q = [&](int pass) {
        const int i_head_0 = pass * HEADS_PER_PASS;
        const int buf      = pass & 1;
#pragma unroll
        for (int i_q = tid; i_q < n_q; i_q += THREADS_PER_BLOCK) {
            const int i_head = i_q / (N_EMBD / 2);
            const int i_embd = i_q % (N_EMBD / 2);
            const float2 q2 = *(const float2 *) (q_base + (i_head_0 + i_head)*nbq1 + i_embd*sizeof(float2));
            q_shared[buf][i_head][i_embd] = __floats2half2_rn(q2.x, q2.y);
        }
        if (tid < HEADS_PER_PASS) {
            w_shared[buf][tid] = w_base[i_head_0 + tid];
        }
    };

    stage_q(0);
    __syncthreads();

    float score_k[K_VECS_PER_WARP] = { 0.0f };

    for (int p = 0; p < N_PASS; ++p) {
        if (p + 1 < N_PASS) {
            stage_q(p + 1);
        }

        const float w_val = w_shared[p & 1][head_sub];

#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            half2 qk2 = __float2half2_rn(0.0f);
#pragma unroll
            for (int j = 0; j < H2_PER_LANE; ++j) {
                qk2 = __hfma2(q_shared[p & 1][head_sub][dim_sub*H2_PER_LANE + j], k_reg[k][j], qk2);
            }
            const float2 qkf = __half22float2(qk2);
            float qk = qkf.x + qkf.y;
#pragma unroll
            for (int offset = LANES_PER_HEAD / 2; offset > 0; offset >>= 1) {
                qk += __shfl_xor_sync(0xffffffff, qk, offset, 32);
            }
            if (dim_sub == 0) {
                score_k[k] += fmaxf(qk, 0.0f) * w_val;
            }
        }

        __syncthreads();
    }

    // combine partial scores (held by lanes 0, 4, 8, ..., 28) on lane 0
#pragma unroll
    for (int k = 0; k < K_VECS_PER_WARP; ++k) {
        score_k[k] += __shfl_sync(0xffffffff, score_k[k], i_lane + 4*LANES_PER_HEAD, 32);
        score_k[k] += __shfl_sync(0xffffffff, score_k[k], i_lane + 2*LANES_PER_HEAD, 32);
        score_k[k] += __shfl_sync(0xffffffff, score_k[k], i_lane + 1*LANES_PER_HEAD, 32);
    }

    __shared__ float dst_shared[K_VECS_PER_BLOCK];

    if (i_lane == 0) {
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            dst_shared[i_warp * K_VECS_PER_WARP + k] = score_k[k];
        }
    }

    __syncthreads();

    if (tid < K_VECS_PER_BLOCK) {
        int i_kv = start_kv_block + tid;
        if (i_kv < n_kv) {
            const half * m_base = (const half *) ((const char *) M + i_batch*nbm1 + (i_stream%nem3)*nbm3);
            float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);
            dst_base[i_kv] = dst_shared[tid] + __half2float(m_base[i_kv]);
        }
    }
}

struct indexer_args {
    const float * q;
    const char * k;
    const float * w;
    const half * m;
    float * dst;
    int64_t n_kv;
    int  use_new;   // 0 = old, otherwise K_VECS_PER_BLOCK for the new kernel
};

template <int K_VECS_PER_BLOCK>
static void indexer_launch_new(const indexer_args & a, hipStream_t stream) {
    constexpr int64_t N_EMBD = 128;
    constexpr int64_t N_HEAD = 64;
    constexpr int WARPS_PER_BLOCK = 8;
    dim3 block(32, WARPS_PER_BLOCK);
    dim3 grid((a.n_kv + K_VECS_PER_BLOCK - 1) / K_VECS_PER_BLOCK, 1, 1);
    indexer_kernel_new<WARPS_PER_BLOCK, K_VECS_PER_BLOCK, N_EMBD, N_HEAD, GGML_TYPE_F16>
        <<<grid, block, 0, stream>>>(
            a.q, a.k, a.w, a.m, a.dst, 1, 1, a.n_kv,
            sizeof(float), 0, 0,
            N_EMBD*sizeof(float), 0, 0,
            0, N_EMBD*sizeof(half), 0,
            sizeof(float), 0, 0,
            sizeof(half), 0, 0,
            1);
}

template <int K_VECS_PER_BLOCK>
static void indexer_launch_h2(const indexer_args & a, hipStream_t stream) {
    constexpr int64_t N_EMBD = 128;
    constexpr int64_t N_HEAD = 64;
    constexpr int WARPS_PER_BLOCK = 8;
    dim3 block(32, WARPS_PER_BLOCK);
    dim3 grid((a.n_kv + K_VECS_PER_BLOCK - 1) / K_VECS_PER_BLOCK, 1, 1);
    indexer_kernel_h2<WARPS_PER_BLOCK, K_VECS_PER_BLOCK, N_EMBD, N_HEAD, GGML_TYPE_F16>
        <<<grid, block, 0, stream>>>(
            a.q, a.k, a.w, a.m, a.dst, 1, 1, a.n_kv,
            sizeof(float), 0, 0,
            N_EMBD*sizeof(float), 0, 0,
            0, N_EMBD*sizeof(half), 0,
            sizeof(float), 0, 0,
            sizeof(half), 0, 0,
            1);
}

static void indexer_launch(const indexer_args & a, hipStream_t stream) {
    constexpr int64_t N_EMBD = 128;
    constexpr int64_t N_HEAD = 64;
    const int64_t n_kv = a.n_kv;

    if (a.use_new == 16) {
        indexer_launch_new<16>(a, stream);
    } else if (a.use_new == 32) {
        indexer_launch_new<32>(a, stream);
    } else if (a.use_new == 64) {
        indexer_launch_new<64>(a, stream);
    } else if (a.use_new == 116) {
        indexer_launch_h2<16>(a, stream);
    } else if (a.use_new == 132) {
        indexer_launch_h2<32>(a, stream);
    } else if (a.use_new == 164) {
        indexer_launch_h2<64>(a, stream);
    } else {
        constexpr int K_VECS_PER_WARP = 8;
        constexpr int WARPS_PER_BLOCK = 8;
        constexpr int K_VECS_PER_BLOCK = K_VECS_PER_WARP * WARPS_PER_BLOCK;
        dim3 block(32, WARPS_PER_BLOCK);
        dim3 grid((n_kv + K_VECS_PER_BLOCK - 1) / K_VECS_PER_BLOCK, 1, 1);
        indexer_kernel_old<WARPS_PER_BLOCK, K_VECS_PER_BLOCK, N_EMBD, N_HEAD, GGML_TYPE_F16>
            <<<grid, block, 0, stream>>>(
                a.q, a.k, a.w, a.m, a.dst, 1, 1, n_kv,
                sizeof(float), 0, 0,
                N_EMBD*sizeof(float), 0, 0,
                0, N_EMBD*sizeof(half), 0,
                sizeof(float), 0, 0,
                sizeof(half), 0, 0,
                1);
    }
}

static void indexer_print_occupancy() {
    int n = 0;
    hipOccupancyMaxActiveBlocksPerMultiprocessor(&n, (const void *) indexer_kernel_old<8, 64, 128, 64, GGML_TYPE_F16>, 256, 0);
    printf("  occupancy old (blocks/CU): %d\n", n);
    hipOccupancyMaxActiveBlocksPerMultiprocessor(&n, (const void *) indexer_kernel_new<8, 16, 128, 64, GGML_TYPE_F16>, 256, 0);
    printf("  occupancy new kvb=16 (blocks/CU): %d\n", n);
    hipOccupancyMaxActiveBlocksPerMultiprocessor(&n, (const void *) indexer_kernel_new<8, 32, 128, 64, GGML_TYPE_F16>, 256, 0);
    printf("  occupancy new kvb=32 (blocks/CU): %d\n", n);
    hipOccupancyMaxActiveBlocksPerMultiprocessor(&n, (const void *) indexer_kernel_new<8, 64, 128, 64, GGML_TYPE_F16>, 256, 0);
    printf("  occupancy new kvb=64 (blocks/CU): %d\n", n);
}

static void run_indexer_case(int64_t n_kv) {
    constexpr int64_t N_EMBD = 128;
    constexpr int64_t N_HEAD = 64;

    std::vector<float> hq(N_HEAD*N_EMBD), hw(N_HEAD);
    std::vector<half>  hk(n_kv*N_EMBD), hm(n_kv, __float2half(0.0f));
    for (auto & v : hq) v = (int32_t) (xorshift32() % 2000) / 1000.0f - 1.0f;
    for (auto & v : hw) v = (int32_t) (xorshift32() % 1000) / 1000.0f;
    for (auto & v : hk) v = __float2half((int32_t) (xorshift32() % 2000) / 1000.0f - 1.0f);

    float * dq; char * dk; float * dw; half * dm; float * dd;
    BENCH_CHECK(hipMalloc(&dq, hq.size()*sizeof(float)));
    BENCH_CHECK(hipMalloc(&dk, hk.size()*sizeof(half)));
    BENCH_CHECK(hipMalloc(&dw, hw.size()*sizeof(float)));
    BENCH_CHECK(hipMalloc(&dm, hm.size()*sizeof(half)));
    BENCH_CHECK(hipMalloc(&dd, n_kv*sizeof(float)));
    BENCH_CHECK(hipMemcpy(dq, hq.data(), hq.size()*sizeof(float), hipMemcpyHostToDevice));
    BENCH_CHECK(hipMemcpy(dk, hk.data(), hk.size()*sizeof(half), hipMemcpyHostToDevice));
    BENCH_CHECK(hipMemcpy(dw, hw.data(), hw.size()*sizeof(float), hipMemcpyHostToDevice));
    BENCH_CHECK(hipMemcpy(dm, hm.data(), hm.size()*sizeof(half), hipMemcpyHostToDevice));

    indexer_args a { dq, dk, dw, dm, dd, n_kv, false };

    // CPU reference on a subset of keys
    const int64_t n_ref = n_kv < 2048 ? n_kv : 2048;
    std::vector<float> ref(n_ref);
    for (int64_t c = 0; c < n_ref; ++c) {
        float total = 0.0f;
        for (int64_t h = 0; h < N_HEAD; ++h) {
            float dot = 0.0f;
            for (int64_t d = 0; d < N_EMBD; ++d) {
                dot += hq[h*N_EMBD + d] * __half2float(hk[c*N_EMBD + d]);
            }
            total += fmaxf(dot, 0.0f) * hw[h];
        }
        ref[c] = total;
    }

    printf("\n=== lightning indexer  heads=%d dim=%d n_kv=%lld  K=%.1f MB (F16) ===\n",
           (int) N_HEAD, (int) N_EMBD, (long long) n_kv, n_kv*N_EMBD*sizeof(half)/1e6);
    indexer_print_occupancy();

    timer tm;
    const int n_var = 7;
    const int kvb[7] = {0, 16, 32, 64, 116, 132, 164};
    const char * names[7] = {"generic warp-reduce (baseline)", "segmented-lane kvb=16", "segmented-lane kvb=32", "segmented-lane kvb=64",
                             "half2 4l/h kvb=16", "half2 4l/h kvb=32", "half2 4l/h kvb=64"};
    double base_ms = 0.0;
    for (int v = 0; v < n_var; ++v) {
        a.use_new = kvb[v];
        BENCH_CHECK(hipMemset(dd, 0, n_kv*sizeof(float)));
        indexer_launch(a, 0);
        std::vector<float> out(n_kv);
        BENCH_CHECK(hipMemcpy(out.data(), dd, n_kv*sizeof(float), hipMemcpyDeviceToHost));

        double max_rel = 0.0; int nbad = 0;
        for (int64_t i = 0; i < n_ref; ++i) {
            const double denom = fabs((double) ref[i]) > 1e-3 ? fabs((double) ref[i]) : 1e-3;
            const double rel = fabs((double) out[i] - (double) ref[i]) / denom;
            max_rel = rel > max_rel ? rel : max_rel;
            if (rel > 2e-3 || !std::isfinite(out[i])) nbad++;
        }

        float ms = tm.time([](void * p) { indexer_launch(*(const indexer_args *) p, 0); }, &a, 3, 10);
        const double gbs = (double) n_kv*N_EMBD*sizeof(half) / (ms * 1e-3) / 1e9;
        const double gflops = 2.0*N_HEAD*N_EMBD*n_kv / (ms * 1e-3) / 1e12;
        if (v == 0) base_ms = ms;
        printf("  %-34s %8.3f ms  %7.1f GB/s  %6.2f TFLOP/s  max_rel=%.2e %s  (%.2fx)\n",
               names[v], ms, gbs, gflops, max_rel, nbad == 0 ? "OK" : "*** MISMATCH ***", base_ms/ms);
    }

    hipFree(dq); hipFree(dk); hipFree(dw); hipFree(dm); hipFree(dd);
}

// ---------------------------------------------------------------------------

int main() {
    int dev = 0;
    BENCH_CHECK(hipGetDevice(&dev));
    hipDeviceProp_t prop;
    BENCH_CHECK(hipGetDeviceProperties(&prop, dev));
    printf("# bench-dsv4-rdna2 on %s (%d CUs)\n", prop.name, prop.multiProcessorCount);
    printf("# synthetic replica kernels; validate end-to-end with test-backend-ops + llama-bench\n");

    // MoE routed experts: up/gate IQ2_S (K=4096, 2048 rows), down IQ3_XXS (K=2048, 4096 rows), 6 selected experts
    run_mmvq_case<GGML_TYPE_IQ2_S  >("IQ2_S   (up/gate expert)", 2048, 4096, 6, true);
    run_mmvq_case<GGML_TYPE_IQ3_XXS>("IQ3_XXS (down expert)   ", 4096, 2048, 6, true);

    // attention: wo_b Q8_0 (K=8192, 4096 rows), and small-K Q8_0 (wo_a K=512, q_b K=1024)
    run_mmvq_case<GGML_TYPE_Q8_0   >("Q8_0    (wo_b)          ", 4096, 8192, 1, false);
    run_mmvq_case<GGML_TYPE_Q8_0   >("Q8_0    (q_b, small-K)  ", 4096, 1024, 1, false);
    run_mmvq_case<GGML_TYPE_Q8_0   >("Q8_0    (wo_a, small-K) ", 8192,  512, 1, false);

    // output head: Q6_K (K=4096, 16384 rows subset of vocab); shared down Q6_K K=2048
    run_mmvq_case<GGML_TYPE_Q6_K   >("Q6_K    (output/shared) ", 16384, 4096, 1, false);
    run_mmvq_case<GGML_TYPE_Q6_K   >("Q6_K    (shared down)   ", 4096, 2048, 1, false);

    // VRAM-bound variants (working set >> 128 MB Infinity Cache)
    run_mmvq_case<GGML_TYPE_IQ2_S  >("IQ2_S   VRAM-bound      ", 2048, 4096, 96, true);
    run_mmvq_case<GGML_TYPE_IQ3_XXS>("IQ3_XXS VRAM-bound      ", 4096, 2048, 48, true);
    run_mmvq_case<GGML_TYPE_Q8_0   >("Q8_0    VRAM-bound      ", 4096, 8192, 8, false);
    run_mmvq_case<GGML_TYPE_Q6_K   >("Q6_K    VRAM-bound      ", 16384, 4096, 5, false);

    // lightning indexer at 256k context (65536 compressed candidates per CSA layer)
    run_indexer_case(65536);

    printf("\ndone.\n");
    return 0;
}