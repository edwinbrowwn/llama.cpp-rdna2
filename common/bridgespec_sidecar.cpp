#include "bridgespec_sidecar.h"
#include "../include/bridgespec/sidecar_abi.h"

#include <cctype>
#include <cstring>
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

using dflash_release_abi_fn = int (*)();
using dflash_check_fn       = int (*)(int32_t, int32_t, int32_t);
using dflash_init_fn        = int (*)(const char *, int32_t);
using dflash_chunk_fn       = int (*)(int32_t, const int32_t *, const float *, int);
using dflash_chunk_device_fn = int (*)(int32_t, const int32_t *, const void * const *, int, int, int);
using dflash_draft_fn       = int (*)(int32_t, int32_t, int32_t, int32_t *);

} // namespace

struct common_bridgespec_mtp_sidecar::impl {
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
};

common_bridgespec_mtp_sidecar::common_bridgespec_mtp_sidecar() : pimpl(new impl) {}

common_bridgespec_mtp_sidecar::~common_bridgespec_mtp_sidecar() {
    // The release ABI currently has no shutdown function.  Keep a successfully
    // initialized library resident until process exit; unloading it would leave
    // its HIP allocations and static state without a supported teardown path.
}

bool common_bridgespec_mtp_sidecar::load(const std::string & library_path,
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
        !resolve_symbol(handle, "spec_hip_draft_device", pimpl->draft_device_fn, error)) {
        close_library(handle);
        return false;
    }
    if (release_abi() != 3) {
        error = "MTP sidecar ABI version mismatch (expected 3)";
        close_library(handle);
        return false;
    }
    if (check(embedding_width, head_rows, n_seq) != 0) {
        error = "MTP sidecar model shape check failed";
        close_library(handle);
        return false;
    }
    if (pimpl->state_size_fn() != static_cast<int>(sizeof(bridgespec_sidecar_state))) {
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

bool common_bridgespec_mtp_sidecar::active() const {
    return pimpl != nullptr && pimpl->active;
}

void common_bridgespec_mtp_sidecar::disable() {
    if (pimpl != nullptr) {
        pimpl->active = false;
    }
}

bool common_bridgespec_mtp_sidecar::get_state(int32_t seq_id, std::vector<uint8_t> & data) const {
    if (!active() || pimpl->state_get_fn == nullptr || pimpl->state_size_fn == nullptr) {
        return false;
    }
    const int size = pimpl->state_size_fn();
    if (size != static_cast<int>(sizeof(bridgespec_sidecar_state))) {
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

bool common_bridgespec_mtp_sidecar::set_state(int32_t seq_id, const std::vector<uint8_t> & data) const {
    return active() && pimpl->state_set_fn != nullptr &&
           data.size() == sizeof(bridgespec_sidecar_state) &&
           data.size() <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
           pimpl->state_set_fn(seq_id, data.data(), static_cast<int>(data.size())) == 0;
}

bool common_bridgespec_mtp_sidecar::reset_state(int32_t seq_id) const {
    return active() && pimpl->state_reset_fn != nullptr && pimpl->state_reset_fn(seq_id) == 0;
}

bool common_bridgespec_mtp_sidecar::truncate_state(int32_t seq_id, int32_t pos_max) const {
    return active() && pimpl->state_truncate_fn != nullptr &&
           pimpl->state_truncate_fn(seq_id, pos_max) == 0;
}

bool common_bridgespec_mtp_sidecar::commit_state(int32_t seq_id, int32_t pos_max, const float * hidden_device) const {
    return active() && pimpl->state_commit_fn != nullptr &&
           pimpl->state_commit_fn(seq_id, pos_max, hidden_device) == 0;
}

bool common_bridgespec_mtp_sidecar::rebase_state(int32_t seq_id, int32_t pos_min, int32_t pos_max, int32_t delta) const {
    return active() && pimpl->state_rebase_fn != nullptr &&
           pimpl->state_rebase_fn(seq_id, pos_min, pos_max, delta) == 0;
}

bool common_bridgespec_mtp_sidecar::attach_target_stream(void * stream, int32_t device) const {
    return active() && pimpl->attach_stream_fn != nullptr &&
           pimpl->attach_stream_fn(stream, device) == 0;
}

int common_bridgespec_mtp_sidecar::catchup(int32_t seq_id, const int32_t * tokens, const int32_t * positions,
        const float * hidden_rows, int count) const {
    return active() && pimpl->catchup_fn != nullptr
        ? pimpl->catchup_fn(seq_id, tokens, positions, hidden_rows, count) : -1;
}

int common_bridgespec_mtp_sidecar::catchup_device(int32_t seq_id, const int32_t * tokens, const int32_t * positions,
        const float * hidden_rows_device, int count) const {
    return active() && pimpl->catchup_device_fn != nullptr
        ? pimpl->catchup_device_fn(seq_id, tokens, positions, hidden_rows_device, count) : -1;
}

int common_bridgespec_mtp_sidecar::draft(int32_t seq_id, int32_t last_token, int32_t past_tokens,
        const float * hidden, int max_draft, int32_t * output_ids) const {
    return active() && pimpl->draft_fn != nullptr
        ? pimpl->draft_fn(seq_id, last_token, past_tokens, hidden, max_draft, output_ids) : -1;
}

int common_bridgespec_mtp_sidecar::draft_device(int32_t seq_id, int32_t last_token, int32_t past_tokens,
        int max_draft, int32_t * output_ids) const {
    return active() && pimpl->draft_device_fn != nullptr
        ? pimpl->draft_device_fn(seq_id, last_token, past_tokens, max_draft, output_ids) : -1;
}

struct common_bridgespec_dflash_sidecar::impl {
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
};

common_bridgespec_dflash_sidecar::common_bridgespec_dflash_sidecar() : pimpl(new impl) {}

common_bridgespec_dflash_sidecar::~common_bridgespec_dflash_sidecar() {
    // See the MTP loader destructor: the current ABI intentionally remains
    // loaded until process exit because it cannot release HIP state.
}

bool common_bridgespec_dflash_sidecar::load(const std::string & library_path,
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
        !resolve_symbol(handle, "spec_dflash_draft", pimpl->draft_fn, error)) {
        close_library(handle);
        return false;
    }
    if (release_abi() != 4) {
        error = "DFlash sidecar ABI version mismatch (expected 4)";
        close_library(handle);
        return false;
    }
    if (check(encoded_width, block_size, n_seq) != 0) {
        error = "DFlash sidecar model shape check failed";
        close_library(handle);
        return false;
    }
    if (pimpl->state_size_fn() != static_cast<int>(sizeof(bridgespec_sidecar_state))) {
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

bool common_bridgespec_dflash_sidecar::active() const {
    return pimpl != nullptr && pimpl->active;
}

void common_bridgespec_dflash_sidecar::disable() {
    if (pimpl != nullptr) {
        pimpl->active = false;
    }
}

bool common_bridgespec_dflash_sidecar::get_state(int32_t seq_id, std::vector<uint8_t> & data) const {
    if (!active() || pimpl->state_get_fn == nullptr || pimpl->state_size_fn == nullptr) {
        return false;
    }
    const int size = pimpl->state_size_fn();
    if (size != static_cast<int>(sizeof(bridgespec_sidecar_state))) {
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

bool common_bridgespec_dflash_sidecar::set_state(int32_t seq_id, const std::vector<uint8_t> & data) const {
    return active() && pimpl->state_set_fn != nullptr &&
           data.size() == sizeof(bridgespec_sidecar_state) &&
           data.size() <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
           pimpl->state_set_fn(seq_id, data.data(), static_cast<int>(data.size())) == 0;
}

bool common_bridgespec_dflash_sidecar::reset_state(int32_t seq_id) const {
    return active() && pimpl->state_reset_fn != nullptr && pimpl->state_reset_fn(seq_id) == 0;
}

bool common_bridgespec_dflash_sidecar::truncate_state(int32_t seq_id, int32_t pos_max) const {
    return active() && pimpl->state_truncate_fn != nullptr &&
           pimpl->state_truncate_fn(seq_id, pos_max) == 0;
}

bool common_bridgespec_dflash_sidecar::commit_state(int32_t seq_id, int32_t pos_max) const {
    return active() && pimpl->state_commit_fn != nullptr &&
           pimpl->state_commit_fn(seq_id, pos_max) == 0;
}

bool common_bridgespec_dflash_sidecar::rebase_state(int32_t seq_id, int32_t pos_min, int32_t pos_max, int32_t delta) const {
    return active() && pimpl->state_rebase_fn != nullptr &&
           pimpl->state_rebase_fn(seq_id, pos_min, pos_max, delta) == 0;
}

bool common_bridgespec_dflash_sidecar::attach_target_stream(void * stream, int32_t device) const {
    return active() && pimpl->attach_stream_fn != nullptr &&
           pimpl->attach_stream_fn(stream, device) == 0;
}

int common_bridgespec_dflash_sidecar::chunk(int32_t seq_id, const int32_t * positions,
        const float * target_features, int count) const {
    return active() && pimpl->chunk_fn != nullptr
        ? pimpl->chunk_fn(seq_id, positions, target_features, count) : -1;
}

int common_bridgespec_dflash_sidecar::chunk_device(int32_t seq_id, const int32_t * positions,
        const void * const * target_layer_features_device, int n_layers, int layer_width, int count) const {
    return active() && pimpl->chunk_device_fn != nullptr
        ? pimpl->chunk_device_fn(seq_id, positions, target_layer_features_device,
                                 n_layers, layer_width, count) : -1;
}

int common_bridgespec_dflash_sidecar::draft(int32_t seq_id, int32_t last_token, int32_t past_tokens,
        int32_t * output_ids) const {
    return active() && pimpl->draft_fn != nullptr
        ? pimpl->draft_fn(seq_id, last_token, past_tokens, output_ids) : -1;
}
