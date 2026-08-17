#include "rccl-tuner-v6.h"

#include <hip/hip_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

struct context {
    bool eligible = false;
    bool applied = false;
    size_t ranks = 0;
    int channels = 0;
};

bool env_present(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0';
}

bool v620_topology(size_t ranks) {
    int count = 0;
    if (ranks < 2 || ranks > 8 || hipGetDeviceCount(&count) != hipSuccess || count < static_cast<int>(ranks) || count > 8) {
        return false;
    }

    for (int i = 0; i < count; ++i) {
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, i) != hipSuccess) {
            return false;
        }
        if (std::strncmp(prop.gcnArchName, "gfx1030", 7) != 0 ||
                std::strncmp(prop.name, "AMD Radeon Pro V620", 20) != 0) {
            return false;
        }
    }
    return true;
}

bool force_mode() {
    const char * mode = std::getenv("GGML_HIP_RCCL_TUNE");
    return mode != nullptr && std::strcmp(mode, "force") == 0;
}

ncclResult_t plugin_init(void ** out, uint64_t, size_t ranks, size_t nodes,
                         ncclDebugLogger_t, ncclNvlDomainInfo_v5_t *, ncclTunerConstants_v6_t *) {
    auto * ctx = new context;
    const bool conflicting_env = env_present("NCCL_ALGO") || env_present("NCCL_PROTO") ||
        env_present("NCCL_MIN_NCHANNELS") || env_present("NCCL_MAX_NCHANNELS") ||
        env_present("NCCL_NTHREADS");
    ctx->ranks = ranks;
    ctx->channels = static_cast<int>(ranks < 3 ? ranks : 3);
    ctx->eligible = force_mode() && nodes == 1 && v620_topology(ranks) && !conflicting_env;

    std::fprintf(stderr, "[rdna2-tuner] v6 loaded mode=%s ranks=%zu nodes=%zu channels=%d eligible=%d\n",
                 std::getenv("GGML_HIP_RCCL_TUNE") ? std::getenv("GGML_HIP_RCCL_TUNE") : "unset",
                 ranks, nodes, ctx->channels, ctx->eligible ? 1 : 0);
    *out = ctx;
    return ncclSuccess;
}

ncclResult_t plugin_get_coll_info(void * opaque, ncclFunc_t coll_type, size_t bytes,
                                  int, float ** costs, int num_algo, int num_proto, int,
                                  int * channels) {
    auto * ctx = static_cast<context *>(opaque);
    if (ctx == nullptr || channels == nullptr || costs == nullptr) {
        return ncclInternalError;
    }

    *channels = 0;
    if (!ctx->eligible || coll_type != ncclFuncAllReduce || bytes != 20480 ||
            num_algo <= NCCL_ALGO_RING || num_proto <= NCCL_PROTO_LL) {
        return ncclSuccess;
    }

    auto table = reinterpret_cast<float (*)[NCCL_NUM_PROTOCOLS_V5]>(costs);
    if (table[NCCL_ALGO_RING][NCCL_PROTO_LL] == NCCL_ALGO_PROTO_IGNORE) {
        return ncclSuccess;
    }

    table[NCCL_ALGO_RING][NCCL_PROTO_LL] = 0.0f;
    *channels = ctx->channels;
    if (!ctx->applied) {
        std::fprintf(stderr, "[rdna2-tuner] applied allreduce bytes=%zu algo=Ring proto=LL channels=%d ranks=%zu\n", bytes, ctx->channels, ctx->ranks);
        ctx->applied = true;
    }
    return ncclSuccess;
}

ncclResult_t plugin_finalize(void * opaque) {
    delete static_cast<context *>(opaque);
    return ncclSuccess;
}

} // namespace

extern "C" const ncclTuner_v6_t ncclTunerPlugin_v6 = {
    "rdna2-v620-hot-size",
    plugin_init,
    plugin_get_coll_info,
    plugin_finalize,
    nullptr,
};
