#include "spec_sidecar.h"
#include "../include/spec_sidecar/sidecar_abi.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <utility>

#ifdef _WIN32
#   ifndef WIN32_LEAN_AND_MEAN
#       define WIN32_LEAN_AND_MEAN
#   endif
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#else
#   include <dlfcn.h>
#endif

namespace {

static bool is_absolute_path(const std::string & path) {
#ifdef _WIN32
    return (path.size() >= 3 && std::isalpha(static_cast<unsigned char>(path[0])) &&
            path[1] == ':' && (path[2] == '\\' || path[2] == '/')) ||
            path.rfind("\\\\", 0) == 0 || path.rfind("//", 0) == 0;
#else
    return !path.empty() && path[0] == '/';
#endif
}

static void * open_library(const std::string & path, std::string & error) {
#ifdef _WIN32
    HMODULE handle = LoadLibraryA(path.c_str());
    if (handle == nullptr) {
        error = "LoadLibrary failed for " + path;
        return nullptr;
    }
    return reinterpret_cast<void *>(handle);
#else
    dlerror();
    void * handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        const char * detail = dlerror();
        error = "dlopen failed for " + path + (detail ? ": " + std::string(detail) : "");
        return nullptr;
    }
    return handle;
#endif
}

static void close_library(void * handle) {
    if (handle == nullptr) {
        return;
    }
#ifdef _WIN32
    FreeLibrary(reinterpret_cast<HMODULE>(handle));
#else
    dlclose(handle);
#endif
}

template<typename Fn>
static bool resolve_symbol(void * handle, const char * name, Fn & result, std::string & error) {
#ifdef _WIN32
    FARPROC symbol = GetProcAddress(reinterpret_cast<HMODULE>(handle), name);
    if (symbol == nullptr) {
        error = std::string("missing sidecar export: ") + name;
        return false;
    }
    static_assert(sizeof(Fn) == sizeof(symbol), "function/data pointer size mismatch");
    std::memcpy(&result, &symbol, sizeof(result));
#else
    dlerror();
    void * symbol = dlsym(handle, name);
    const char * detail = dlerror();
    if (detail != nullptr || symbol == nullptr) {
        error = std::string("missing sidecar export: ") + name;
        return false;
    }
    static_assert(sizeof(Fn) == sizeof(symbol), "function/data pointer size mismatch");
    std::memcpy(&result, &symbol, sizeof(result));
#endif
    return true;
}

static bool require_absolute(const std::string & path, const char * label, std::string & error) {
    if (!is_absolute_path(path)) {
        error = std::string(label) + " must be an absolute path";
        return false;
    }
    return true;
}

static bool require_file(const std::string & path, const char * label, std::string & error) {
    std::ifstream file(path, std::ios::binary);
    if (!file.good()) {
        error = std::string(label) + " is not readable: " + path;
        return false;
    }
    return true;
}

static std::string join_path(const std::string & dir, const char * name) {
#ifdef _WIN32
    const char separator = '\\';
#else
    const char separator = '/';
#endif
    return dir + separator + name;
}

using state_size_fn_t     = int (*)();
using state_get_fn_t      = int (*)(int32_t, void *, int);
using state_set_fn_t      = int (*)(int32_t, const void *, int);
using state_reset_fn_t    = int (*)(int32_t);
using state_truncate_fn_t = int (*)(int32_t, int32_t);
using mtp_state_commit_fn_t = int (*)(int32_t, int32_t, const float *);
using state_commit_fn_t      = int (*)(int32_t, int32_t);
using state_rebase_fn_t      = int (*)(int32_t, int32_t, int32_t, int32_t);
using attach_stream_fn_t     = int (*)(void *, int32_t);

using mtp_release_abi_fn = int (*)();
using mtp_check_fn      = int (*)(int32_t, int32_t, int32_t);
using mtp_init_fn       = int (*)(const char *, const char *, int32_t);
using mtp_catchup_fn    = int (*)(int32_t, const int32_t *, const int32_t *, const float *, int);
using mtp_catchup_device_fn = int (*)(int32_t, const int32_t *, const int32_t *, const float *, int);
using mtp_draft_fn      = int (*)(int32_t, int32_t, int32_t, const float *, int, int32_t *);
using mtp_draft_device_fn = int (*)(int32_t, int32_t, int32_t, int, int32_t *);
using mtp_stochastic_top_k_fn = int (*)();
using mtp_draft_stochastic_fn = int (*)(int32_t, int32_t, int32_t, const float *, float, float, uint64_t, int, int32_t *, int32_t *, float *);
using mtp_draft_stochastic_device_fn = int (*)(int32_t, int32_t, int32_t, float, float, uint64_t, int, int32_t *, int32_t *, float *);

using dflash_release_abi_fn = int (*)();
using dflash_check_fn       = int (*)(int32_t, int32_t, int32_t);
using dflash_init_fn        = int (*)(const char *, int32_t);
using dflash_chunk_fn       = int (*)(int32_t, const int32_t *, const float *, int);
using dflash_chunk_device_fn = int (*)(int32_t, const int32_t *, const void * const *, int, int, int);
using dflash_draft_fn       = int (*)(int32_t, int32_t, int32_t, int32_t *);
using dflash_stochastic_top_k_fn = int (*)();
using dflash_draft_stochastic_fn = int (*)(int32_t, int32_t, int32_t, float, float, uint64_t, int, int32_t *, int32_t *, float *);

} // namespace

bool common_spec_sidecar_mtp_probe(const std::string & library_path,
        const std::string & weights_dir, const std::string & ids_path,
        int32_t embedding_width, int32_t head_rows, int32_t n_seq,
        std::string & error) {
    if (n_seq < 1 || n_seq > 8) {
        error = "MTP sidecar supports 1..8 sequences";
        return false;
    }
    if (!require_absolute(library_path, "MTP sidecar library path", error) ||
        !require_absolute(weights_dir, "MTP sidecar artifact directory", error) ||
        !require_absolute(ids_path, "MTP sidecar ID path", error) ||
        !require_file(join_path(weights_dir, "drafter_manifest.json"),
                "MTP sidecar manifest", error) ||
        !require_file(join_path(weights_dir, "drafter_weights.bin"),
                "MTP sidecar weights", error) ||
        !require_file(ids_path, "MTP sidecar ID table", error)) {
        return false;
    }

    void * handle = open_library(library_path, error);
    if (handle == nullptr) {
        return false;
    }
    mtp_release_abi_fn release = nullptr;
    mtp_check_fn check = nullptr;
    mtp_stochastic_top_k_fn top_k = nullptr;
    mtp_draft_stochastic_fn stochastic = nullptr;
    mtp_draft_stochastic_device_fn stochastic_device = nullptr;
    const bool symbols =
        resolve_symbol(handle, "spec_hip_release_abi", release, error) &&
        resolve_symbol(handle, "spec_hip_check", check, error) &&
        resolve_symbol(handle, "spec_hip_stochastic_top_k", top_k, error) &&
        resolve_symbol(handle, "spec_hip_draft_stochastic", stochastic, error) &&
        resolve_symbol(handle, "spec_hip_draft_stochastic_device", stochastic_device, error);
    const bool compatible = symbols && release() == 4 &&
            check(embedding_width, head_rows, n_seq) == 0 &&
            top_k() == SPEC_SIDECAR_MTP_DRAFT_TOP_K;
    if (!compatible && error.empty()) {
        error = "MTP sidecar stochastic ABI probe failed";
    }
    close_library(handle);
    return compatible;
}

bool common_spec_sidecar_dflash_probe(const std::string & library_path,
        const std::string & artifact_dir, int32_t encoded_width,
        int32_t block_size, int32_t n_seq, std::string & error) {
    if (n_seq < 1 || n_seq > 8) {
        error = "DFlash sidecar supports 1..8 sequences";
        return false;
    }
    if (!require_absolute(library_path, "DFlash sidecar library path", error) ||
        !require_absolute(artifact_dir, "DFlash sidecar artifact directory", error) ||
        !require_file(join_path(artifact_dir, "dflash_manifest.json"),
                "DFlash sidecar manifest", error) ||
        !require_file(join_path(artifact_dir, "dflash_weights.bin"),
                "DFlash sidecar weights", error) ||
        !require_file(join_path(artifact_dir,
                std::getenv("LLAMA_SPEC_HIP_FULL_HEAD") != nullptr
                    ? "target_head.bin" : "target_head_sliced.bin"),
                "DFlash target head", error) ||
        (std::getenv("LLAMA_SPEC_HIP_FULL_HEAD") == nullptr &&
         !require_file(join_path(artifact_dir, "draft_head_ids.bin"),
                "DFlash target-head ID table", error)) ||
        !require_file(join_path(artifact_dir, "drafter_manifest.json"),
                "DFlash target embedding manifest", error) ||
        !require_file(join_path(artifact_dir, "drafter_weights.bin"),
                "DFlash target embedding", error)) {
        return false;
    }

    void * handle = open_library(library_path, error);
    if (handle == nullptr) {
        return false;
    }
    dflash_release_abi_fn release = nullptr;
    dflash_check_fn check = nullptr;
    dflash_stochastic_top_k_fn top_k = nullptr;
    dflash_draft_stochastic_fn stochastic = nullptr;
    const bool symbols =
        resolve_symbol(handle, "spec_dflash_release_abi", release, error) &&
        resolve_symbol(handle, "spec_dflash_check", check, error) &&
        resolve_symbol(handle, "spec_dflash_stochastic_top_k", top_k, error) &&
        resolve_symbol(handle, "spec_dflash_draft_stochastic", stochastic, error);
    const bool compatible = symbols && release() == 5 &&
            check(encoded_width, block_size, n_seq) == 0 &&
            top_k() == SPEC_SIDECAR_DFLASH_DRAFT_TOP_K;
    if (!compatible && error.empty()) {
        error = "DFlash sidecar stochastic ABI probe failed";
    }
    close_library(handle);
    return compatible;
}

struct common_spec_sidecar_mtp::impl {
    void * handle = nullptr;
    bool active = false;
    state_size_fn_t state_size_fn = nullptr;
    state_get_fn_t state_get_fn = nullptr;
    state_set_fn_t state_set_fn = nullptr;
    state_reset_fn_t state_reset_fn = nullptr;
    state_truncate_fn_t state_truncate_fn = nullptr;
    mtp_state_commit_fn_t state_commit_fn = nullptr;
    state_rebase_fn_t state_rebase_fn = nullptr;
    attach_stream_fn_t attach_stream_fn = nullptr;
    mtp_catchup_fn catchup_fn = nullptr;
    mtp_catchup_device_fn catchup_device_fn = nullptr;
    mtp_draft_fn draft_fn = nullptr;
    mtp_draft_device_fn draft_device_fn = nullptr;
    mtp_stochastic_top_k_fn stochastic_top_k_fn = nullptr;
    mtp_draft_stochastic_fn draft_stochastic_fn = nullptr;
    mtp_draft_stochastic_device_fn draft_stochastic_device_fn = nullptr;
};

common_spec_sidecar_mtp::common_spec_sidecar_mtp() : pimpl(new impl) {}

common_spec_sidecar_mtp::~common_spec_sidecar_mtp() {
    // The release ABI currently has no shutdown function.  Keep a successfully
    // initialized library resident until process exit; unloading it would leave
    // its HIP allocations and static state without a supported teardown path.
}

bool common_spec_sidecar_mtp::load(const std::string & library_path,
        const std::string & weights_dir, const std::string & ids_path,
        int32_t embedding_width, int32_t head_rows, int32_t n_seq, std::string & error) {
    if (active()) {
        error = "MTP sidecar is already loaded";
        return false;
    }
    if (n_seq < 1 || n_seq > 8) {
        error = "MTP sidecar supports 1..8 sequences";
        return false;
    }
    if (!require_absolute(library_path, "MTP sidecar library path", error) ||
        !require_absolute(weights_dir, "MTP sidecar artifact directory", error) ||
        !require_absolute(ids_path, "MTP sidecar ID path", error)) {
        return false;
    }

    void * handle = open_library(library_path, error);
    if (handle == nullptr) {
        return false;
    }

    mtp_release_abi_fn release_abi = nullptr;
    mtp_check_fn check = nullptr;
    mtp_init_fn init = nullptr;
    if (!resolve_symbol(handle, "spec_hip_release_abi", release_abi, error) ||
        !resolve_symbol(handle, "spec_hip_check", check, error) ||
        !resolve_symbol(handle, "spec_hip_state_size", pimpl->state_size_fn, error) ||
        !resolve_symbol(handle, "spec_hip_get_state", pimpl->state_get_fn, error) ||
        !resolve_symbol(handle, "spec_hip_set_state", pimpl->state_set_fn, error) ||
        !resolve_symbol(handle, "spec_hip_reset_state", pimpl->state_reset_fn, error) ||
        !resolve_symbol(handle, "spec_hip_truncate_state", pimpl->state_truncate_fn, error) ||
        !resolve_symbol(handle, "spec_hip_commit_state", pimpl->state_commit_fn, error) ||
        !resolve_symbol(handle, "spec_hip_rebase_state", pimpl->state_rebase_fn, error) ||
        !resolve_symbol(handle, "spec_hip_attach_target_stream", pimpl->attach_stream_fn, error) ||
        !resolve_symbol(handle, "spec_hip_init", init, error) ||
        !resolve_symbol(handle, "spec_hip_catchup", pimpl->catchup_fn, error) ||
        !resolve_symbol(handle, "spec_hip_catchup_device", pimpl->catchup_device_fn, error) ||
        !resolve_symbol(handle, "spec_hip_draft", pimpl->draft_fn, error) ||
        !resolve_symbol(handle, "spec_hip_draft_device", pimpl->draft_device_fn, error) ||
        !resolve_symbol(handle, "spec_hip_stochastic_top_k", pimpl->stochastic_top_k_fn, error) ||
        !resolve_symbol(handle, "spec_hip_draft_stochastic", pimpl->draft_stochastic_fn, error) ||
        !resolve_symbol(handle, "spec_hip_draft_stochastic_device", pimpl->draft_stochastic_device_fn, error)) {
        close_library(handle);
        return false;
    }
    if (release_abi() != 4) {
        error = "MTP sidecar ABI version mismatch (expected 4)";
        close_library(handle);
        return false;
    }
    if (check(embedding_width, head_rows, n_seq) != 0) {
        error = "MTP sidecar model shape check failed";
        close_library(handle);
        return false;
    }
    if (pimpl->stochastic_top_k_fn() != SPEC_SIDECAR_MTP_DRAFT_TOP_K) {
        error = "MTP sidecar stochastic top-k mismatch";
        close_library(handle);
        return false;
    }
    if (pimpl->state_size_fn() != static_cast<int>(sizeof(spec_sidecar_state))) {
        error = "MTP sidecar state ABI size mismatch";
        close_library(handle);
        pimpl->state_size_fn = nullptr;
        pimpl->state_get_fn = nullptr;
        pimpl->state_set_fn = nullptr;
        pimpl->state_reset_fn = nullptr;
        pimpl->state_truncate_fn = nullptr;
        pimpl->state_rebase_fn = nullptr;
        pimpl->catchup_fn = nullptr;
        pimpl->draft_fn = nullptr;
        return false;
    }
    if (init(weights_dir.c_str(), ids_path.c_str(), n_seq) != 0) {
        error = "MTP sidecar initialization failed";
        close_library(handle);
        pimpl->state_size_fn = nullptr;
        pimpl->state_get_fn = nullptr;
        pimpl->state_set_fn = nullptr;
        pimpl->state_reset_fn = nullptr;
        pimpl->state_truncate_fn = nullptr;
        pimpl->state_rebase_fn = nullptr;
        pimpl->catchup_fn = nullptr;
        pimpl->draft_fn = nullptr;
        return false;
    }

    pimpl->handle = handle;
    pimpl->active = true;
    return true;
}

bool common_spec_sidecar_mtp::active() const {
    return pimpl != nullptr && pimpl->active;
}

void common_spec_sidecar_mtp::disable() {
    if (pimpl != nullptr) {
        pimpl->active = false;
    }
}

bool common_spec_sidecar_mtp::get_state(int32_t seq_id, std::vector<uint8_t> & data) const {
    if (!active() || pimpl->state_get_fn == nullptr || pimpl->state_size_fn == nullptr) {
        return false;
    }
    const int size = pimpl->state_size_fn();
    if (size != static_cast<int>(sizeof(spec_sidecar_state))) {
        data.clear();
        return false;
    }
    data.resize(static_cast<size_t>(size));
    if (pimpl->state_get_fn(seq_id, data.data(), size) != 0) {
        data.clear();
        return false;
    }
    return true;
}

bool common_spec_sidecar_mtp::set_state(int32_t seq_id, const std::vector<uint8_t> & data) const {
    return active() && pimpl->state_set_fn != nullptr &&
           data.size() == sizeof(spec_sidecar_state) &&
           data.size() <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
           pimpl->state_set_fn(seq_id, data.data(), static_cast<int>(data.size())) == 0;
}

bool common_spec_sidecar_mtp::reset_state(int32_t seq_id) const {
    return active() && pimpl->state_reset_fn != nullptr && pimpl->state_reset_fn(seq_id) == 0;
}

bool common_spec_sidecar_mtp::truncate_state(int32_t seq_id, int32_t pos_max) const {
    return active() && pimpl->state_truncate_fn != nullptr &&
           pimpl->state_truncate_fn(seq_id, pos_max) == 0;
}

bool common_spec_sidecar_mtp::commit_state(int32_t seq_id, int32_t pos_max, const float * hidden_device) const {
    return active() && pimpl->state_commit_fn != nullptr &&
           pimpl->state_commit_fn(seq_id, pos_max, hidden_device) == 0;
}

bool common_spec_sidecar_mtp::rebase_state(int32_t seq_id, int32_t pos_min, int32_t pos_max, int32_t delta) const {
    return active() && pimpl->state_rebase_fn != nullptr &&
           pimpl->state_rebase_fn(seq_id, pos_min, pos_max, delta) == 0;
}

bool common_spec_sidecar_mtp::attach_target_stream(void * stream, int32_t device) const {
    return active() && pimpl->attach_stream_fn != nullptr &&
           pimpl->attach_stream_fn(stream, device) == 0;
}

int common_spec_sidecar_mtp::catchup(int32_t seq_id, const int32_t * tokens, const int32_t * positions,
        const float * hidden_rows, int count) const {
    return active() && pimpl->catchup_fn != nullptr
        ? pimpl->catchup_fn(seq_id, tokens, positions, hidden_rows, count) : -1;
}

int common_spec_sidecar_mtp::catchup_device(int32_t seq_id, const int32_t * tokens, const int32_t * positions,
        const float * hidden_rows_device, int count) const {
    return active() && pimpl->catchup_device_fn != nullptr
        ? pimpl->catchup_device_fn(seq_id, tokens, positions, hidden_rows_device, count) : -1;
}

int common_spec_sidecar_mtp::draft(int32_t seq_id, int32_t last_token, int32_t past_tokens,
        const float * hidden, int max_draft, int32_t * output_ids) const {
    return active() && pimpl->draft_fn != nullptr
        ? pimpl->draft_fn(seq_id, last_token, past_tokens, hidden, max_draft, output_ids) : -1;
}

int common_spec_sidecar_mtp::draft_device(int32_t seq_id, int32_t last_token, int32_t past_tokens,
        int max_draft, int32_t * output_ids) const {
    return active() && pimpl->draft_device_fn != nullptr
        ? pimpl->draft_device_fn(seq_id, last_token, past_tokens, max_draft, output_ids) : -1;
}

int common_spec_sidecar_mtp::draft_stochastic(int32_t seq_id, int32_t last_token,
        int32_t past_tokens, const float * hidden, float temperature, float p_min,
        uint64_t rng_key, int max_draft, int32_t * output_ids,
        int32_t * dist_ids, float * dist_probs) const {
    return active() && pimpl->draft_stochastic_fn != nullptr
        ? pimpl->draft_stochastic_fn(seq_id, last_token, past_tokens, hidden,
                temperature, p_min, rng_key, max_draft, output_ids, dist_ids, dist_probs) : -1;
}

int common_spec_sidecar_mtp::draft_stochastic_device(int32_t seq_id, int32_t last_token,
        int32_t past_tokens, float temperature, float p_min, uint64_t rng_key,
        int max_draft, int32_t * output_ids, int32_t * dist_ids, float * dist_probs) const {
    return active() && pimpl->draft_stochastic_device_fn != nullptr
        ? pimpl->draft_stochastic_device_fn(seq_id, last_token, past_tokens,
                temperature, p_min, rng_key, max_draft, output_ids, dist_ids, dist_probs) : -1;
}

struct common_spec_sidecar_dflash::impl {
    void * handle = nullptr;
    bool active = false;
    state_size_fn_t state_size_fn = nullptr;
    state_get_fn_t state_get_fn = nullptr;
    state_set_fn_t state_set_fn = nullptr;
    state_reset_fn_t state_reset_fn = nullptr;
    state_truncate_fn_t state_truncate_fn = nullptr;
    state_commit_fn_t state_commit_fn = nullptr;
    state_rebase_fn_t state_rebase_fn = nullptr;
    attach_stream_fn_t attach_stream_fn = nullptr;
    dflash_chunk_fn chunk_fn = nullptr;
    dflash_chunk_device_fn chunk_device_fn = nullptr;
    dflash_draft_fn draft_fn = nullptr;
    dflash_stochastic_top_k_fn stochastic_top_k_fn = nullptr;
    dflash_draft_stochastic_fn draft_stochastic_fn = nullptr;
};

common_spec_sidecar_dflash::common_spec_sidecar_dflash() : pimpl(new impl) {}

common_spec_sidecar_dflash::~common_spec_sidecar_dflash() {
    // See the MTP loader destructor: the current ABI intentionally remains
    // loaded until process exit because it cannot release HIP state.
}

bool common_spec_sidecar_dflash::load(const std::string & library_path,
        const std::string & artifact_dir, int32_t encoded_width, int32_t block_size,
        int32_t n_seq, std::string & error) {
    if (active()) {
        error = "DFlash sidecar is already loaded";
        return false;
    }
    if (n_seq < 1 || n_seq > 8) {
        error = "DFlash sidecar supports 1..8 sequences";
        return false;
    }
    if (!require_absolute(library_path, "DFlash sidecar library path", error) ||
        !require_absolute(artifact_dir, "DFlash sidecar artifact directory", error)) {
        return false;
    }

    void * handle = open_library(library_path, error);
    if (handle == nullptr) {
        return false;
    }

    dflash_release_abi_fn release_abi = nullptr;
    dflash_check_fn check = nullptr;
    dflash_init_fn init = nullptr;
    if (!resolve_symbol(handle, "spec_dflash_release_abi", release_abi, error) ||
        !resolve_symbol(handle, "spec_dflash_check", check, error) ||
        !resolve_symbol(handle, "spec_dflash_state_size", pimpl->state_size_fn, error) ||
        !resolve_symbol(handle, "spec_dflash_get_state", pimpl->state_get_fn, error) ||
        !resolve_symbol(handle, "spec_dflash_set_state", pimpl->state_set_fn, error) ||
        !resolve_symbol(handle, "spec_dflash_reset_state", pimpl->state_reset_fn, error) ||
        !resolve_symbol(handle, "spec_dflash_truncate_state", pimpl->state_truncate_fn, error) ||
        !resolve_symbol(handle, "spec_dflash_commit_state", pimpl->state_commit_fn, error) ||
        !resolve_symbol(handle, "spec_dflash_rebase_state", pimpl->state_rebase_fn, error) ||
        !resolve_symbol(handle, "spec_dflash_attach_target_stream", pimpl->attach_stream_fn, error) ||
        !resolve_symbol(handle, "spec_dflash_init", init, error) ||
        !resolve_symbol(handle, "spec_dflash_chunk", pimpl->chunk_fn, error) ||
        !resolve_symbol(handle, "spec_dflash_chunk_device", pimpl->chunk_device_fn, error) ||
        !resolve_symbol(handle, "spec_dflash_draft", pimpl->draft_fn, error) ||
        !resolve_symbol(handle, "spec_dflash_stochastic_top_k", pimpl->stochastic_top_k_fn, error) ||
        !resolve_symbol(handle, "spec_dflash_draft_stochastic", pimpl->draft_stochastic_fn, error)) {
        close_library(handle);
        return false;
    }
    if (release_abi() != 5) {
        error = "DFlash sidecar ABI version mismatch (expected 5)";
        close_library(handle);
        return false;
    }
    if (check(encoded_width, block_size, n_seq) != 0) {
        error = "DFlash sidecar model shape check failed";
        close_library(handle);
        return false;
    }
    if (pimpl->stochastic_top_k_fn() != SPEC_SIDECAR_DFLASH_DRAFT_TOP_K) {
        error = "DFlash sidecar stochastic top-k mismatch";
        close_library(handle);
        return false;
    }
    if (pimpl->state_size_fn() != static_cast<int>(sizeof(spec_sidecar_state))) {
        error = "DFlash sidecar state ABI size mismatch";
        close_library(handle);
        pimpl->state_size_fn = nullptr;
        pimpl->state_get_fn = nullptr;
        pimpl->state_set_fn = nullptr;
        pimpl->state_reset_fn = nullptr;
        pimpl->state_truncate_fn = nullptr;
        pimpl->state_rebase_fn = nullptr;
        pimpl->chunk_fn = nullptr;
        pimpl->draft_fn = nullptr;
        return false;
    }
    if (init(artifact_dir.c_str(), n_seq) != 0) {
        error = "DFlash sidecar initialization failed";
        close_library(handle);
        pimpl->state_size_fn = nullptr;
        pimpl->state_get_fn = nullptr;
        pimpl->state_set_fn = nullptr;
        pimpl->state_reset_fn = nullptr;
        pimpl->state_truncate_fn = nullptr;
        pimpl->state_rebase_fn = nullptr;
        pimpl->chunk_fn = nullptr;
        pimpl->draft_fn = nullptr;
        return false;
    }

    pimpl->handle = handle;
    pimpl->active = true;
    return true;
}

bool common_spec_sidecar_dflash::active() const {
    return pimpl != nullptr && pimpl->active;
}

void common_spec_sidecar_dflash::disable() {
    if (pimpl != nullptr) {
        pimpl->active = false;
    }
}

bool common_spec_sidecar_dflash::get_state(int32_t seq_id, std::vector<uint8_t> & data) const {
    if (!active() || pimpl->state_get_fn == nullptr || pimpl->state_size_fn == nullptr) {
        return false;
    }
    const int size = pimpl->state_size_fn();
    if (size != static_cast<int>(sizeof(spec_sidecar_state))) {
        data.clear();
        return false;
    }
    data.resize(static_cast<size_t>(size));
    if (pimpl->state_get_fn(seq_id, data.data(), size) != 0) {
        data.clear();
        return false;
    }
    return true;
}

bool common_spec_sidecar_dflash::set_state(int32_t seq_id, const std::vector<uint8_t> & data) const {
    return active() && pimpl->state_set_fn != nullptr &&
           data.size() == sizeof(spec_sidecar_state) &&
           data.size() <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
           pimpl->state_set_fn(seq_id, data.data(), static_cast<int>(data.size())) == 0;
}

bool common_spec_sidecar_dflash::reset_state(int32_t seq_id) const {
    return active() && pimpl->state_reset_fn != nullptr && pimpl->state_reset_fn(seq_id) == 0;
}

bool common_spec_sidecar_dflash::truncate_state(int32_t seq_id, int32_t pos_max) const {
    return active() && pimpl->state_truncate_fn != nullptr &&
           pimpl->state_truncate_fn(seq_id, pos_max) == 0;
}

bool common_spec_sidecar_dflash::commit_state(int32_t seq_id, int32_t pos_max) const {
    return active() && pimpl->state_commit_fn != nullptr &&
           pimpl->state_commit_fn(seq_id, pos_max) == 0;
}

bool common_spec_sidecar_dflash::rebase_state(int32_t seq_id, int32_t pos_min, int32_t pos_max, int32_t delta) const {
    return active() && pimpl->state_rebase_fn != nullptr &&
           pimpl->state_rebase_fn(seq_id, pos_min, pos_max, delta) == 0;
}

bool common_spec_sidecar_dflash::attach_target_stream(void * stream, int32_t device) const {
    return active() && pimpl->attach_stream_fn != nullptr &&
           pimpl->attach_stream_fn(stream, device) == 0;
}

int common_spec_sidecar_dflash::chunk(int32_t seq_id, const int32_t * positions,
        const float * target_features, int count) const {
    return active() && pimpl->chunk_fn != nullptr
        ? pimpl->chunk_fn(seq_id, positions, target_features, count) : -1;
}

int common_spec_sidecar_dflash::chunk_device(int32_t seq_id, const int32_t * positions,
        const void * const * target_layer_features_device, int n_layers, int layer_width, int count) const {
    return active() && pimpl->chunk_device_fn != nullptr
        ? pimpl->chunk_device_fn(seq_id, positions, target_layer_features_device,
                                 n_layers, layer_width, count) : -1;
}

int common_spec_sidecar_dflash::draft(int32_t seq_id, int32_t last_token, int32_t past_tokens,
        int32_t * output_ids) const {
    return active() && pimpl->draft_fn != nullptr
        ? pimpl->draft_fn(seq_id, last_token, past_tokens, output_ids) : -1;
}

int common_spec_sidecar_dflash::draft_stochastic(int32_t seq_id, int32_t last_token,
        int32_t past_tokens, float temperature, float p_min, uint64_t rng_key,
        int max_draft, int32_t * output_ids, int32_t * dist_ids, float * dist_probs) const {
    return active() && pimpl->draft_stochastic_fn != nullptr
        ? pimpl->draft_stochastic_fn(seq_id, last_token, past_tokens, temperature,
                p_min, rng_key, max_draft, output_ids, dist_ids, dist_probs) : -1;
}
