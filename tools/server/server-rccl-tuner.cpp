#include "server-rccl-tuner.h"

#include "log.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

#if defined(__linux__)
#include <unistd.h>
#endif

namespace {

namespace fs = std::filesystem;

bool parse_mode(const char * value, common_rccl_tune_mode & mode) {
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "auto") == 0) {
        mode = COMMON_RCCL_TUNE_AUTO;
        return true;
    }
    if (std::strcmp(value, "off") == 0) {
        mode = COMMON_RCCL_TUNE_OFF;
        return true;
    }
    if (std::strcmp(value, "force") == 0) {
        mode = COMMON_RCCL_TUNE_FORCE;
        return true;
    }
    return false;
}

bool env_present(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0';
}

bool tensor_split_is_four_way(const common_params & params) {
    size_t nonzero = 0;
    for (size_t i = 0; i < 128; ++i) {
        if (params.tensor_split[i] > 0.0f) {
            ++nonzero;
        }
    }
    return nonzero == 4;
}

bool requested_shape_is_eligible(const common_params & params) {
    if (params.split_mode != LLAMA_SPLIT_MODE_TENSOR || !tensor_split_is_four_way(params)) {
        return false;
    }

    if (!params.devices.empty()) {
        size_t n_devices = 0;
        for (auto * device : params.devices) {
            if (device != nullptr) {
                ++n_devices;
            }
        }
        if (n_devices != 4) {
            return false;
        }
    }
    return true;
}

fs::path executable_path(const char * argv0) {
#if defined(__linux__)
    char buffer[4096] = {};
    const ssize_t n = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (n > 0) {
        return fs::path(std::string(buffer, static_cast<size_t>(n)));
    }
#endif
    if (argv0 != nullptr && argv0[0] != '\0') {
        return fs::absolute(fs::path(argv0));
    }
    return {};
}

fs::path find_plugin(const char * argv0) {
    if (const char * override_path = std::getenv("GGML_HIP_RCCL_PLUGIN");
            override_path != nullptr && override_path[0] != '\0') {
        return fs::path(override_path);
    }

    const fs::path exe = executable_path(argv0);
    if (exe.empty()) {
        return {};
    }

    const fs::path alongside = exe.parent_path() / "libnccl-tuner-rdna2-v620.so";
    if (fs::is_regular_file(alongside)) {
        return alongside;
    }

    const fs::path libdir = exe.parent_path().parent_path() / "lib" / "libnccl-tuner-rdna2-v620.so";
    if (fs::is_regular_file(libdir)) {
        return libdir;
    }
    return {};
}

bool has_user_collective_override() {
    static constexpr const char * names[] = {
        "NCCL_ALGO", "NCCL_PROTO", "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS",
        "NCCL_NTHREADS", "NCCL_TUNER_PLUGIN",
    };
    for (const char * name : names) {
        if (env_present(name)) {
            LOG_WRN("RCCL native tuner: %s is user-set; leaving RCCL Auto unchanged\n", name);
            return true;
        }
    }
    return false;
}

} // namespace

void server_rccl_tuner_prepare(const common_params & params, const char * argv0) {
#if !defined(__linux__)
    GGML_UNUSED(params);
    GGML_UNUSED(argv0);
    return;
#else
    common_rccl_tune_mode mode = params.rccl_tune;
    if (const char * env_mode = std::getenv("GGML_HIP_RCCL_TUNE"); env_mode != nullptr) {
        if (!parse_mode(env_mode, mode)) {
            LOG_WRN("RCCL native tuner: invalid GGML_HIP_RCCL_TUNE=%s; using Auto\n", env_mode);
            mode = COMMON_RCCL_TUNE_OFF;
        }
    }

    if (mode == COMMON_RCCL_TUNE_OFF) {
        unsetenv("GGML_HIP_RCCL_TUNE");
        LOG_INF("RCCL native tuner: off; using RCCL Auto\n");
        return;
    }

    const char * allreduce = std::getenv("GGML_CUDA_ALLREDUCE");
    if (allreduce != nullptr && std::strcmp(allreduce, "nccl") != 0) {
        LOG_INF("RCCL native tuner: GGML_CUDA_ALLREDUCE=%s; using configured backend\n", allreduce);
        return;
    }

    if (has_user_collective_override()) {
        return;
    }

    if (!requested_shape_is_eligible(params)) {
        LOG_INF("RCCL native tuner: non-TP4 tensor-split shape; using RCCL Auto\n");
        return;
    }

    const fs::path plugin = find_plugin(argv0);
    if (plugin.empty()) {
        LOG_WRN("RCCL native tuner: plugin not found beside llama-server; using RCCL Auto\n");
        return;
    }

    if (mode == COMMON_RCCL_TUNE_FORCE) {
        setenv("GGML_HIP_RCCL_TUNE", "force", 1);
        setenv("NCCL_TUNER_PLUGIN", plugin.c_str(), 1);
        LOG_INF("RCCL native tuner: force enabled; policy frozen before communicator initialization\n");
        return;
    }

    // Auto uses the offline-certified policy only after the strict launch-shape
    // gate above.  RCCL's plugin performs the final four-V620 topology check;
    // all unsupported cases return an untouched cost table and remain Auto.
    setenv("GGML_HIP_RCCL_TUNE", "force", 1);
    setenv("NCCL_TUNER_PLUGIN", plugin.c_str(), 1);
    LOG_INF("RCCL native tuner: auto selected offline-certified Ring/LL/3-hot policy; policy frozen before communicator initialization\n");
#endif
}
