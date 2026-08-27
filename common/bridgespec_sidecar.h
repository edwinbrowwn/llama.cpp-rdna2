#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Host-side loader for the optional BridgeSpec Qwen3.8-27B sidecars.
// The sidecars are deliberately opt-in and model-specific.  A loader object
// keeps the library resident for the lifetime of the process because the
// current release ABI has no shutdown operation.
class common_bridgespec_mtp_sidecar {
public:
    common_bridgespec_mtp_sidecar();
    ~common_bridgespec_mtp_sidecar();

    common_bridgespec_mtp_sidecar(const common_bridgespec_mtp_sidecar &) = delete;
    common_bridgespec_mtp_sidecar & operator=(const common_bridgespec_mtp_sidecar &) = delete;

    bool load(const std::string & library_path,
              const std::string & weights_dir,
              const std::string & ids_path,
              int32_t embedding_width,
              int32_t head_rows,
              std::string & error);

    bool active() const;
    void disable();

    bool get_state(std::vector<uint8_t> & data) const;
    bool set_state(const std::vector<uint8_t> & data) const;
    bool reset_state() const;
    bool truncate_state(int32_t pos_max) const;
    bool rebase_state(int32_t pos_min, int32_t pos_max, int32_t delta) const;

    int catchup(const int32_t * tokens, const int32_t * positions,
                const float * hidden_rows, int count) const;
    int draft(int32_t last_token, int32_t past_tokens,
              const float * hidden, int max_draft, int32_t * output_ids) const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

class common_bridgespec_dflash_sidecar {
public:
    common_bridgespec_dflash_sidecar();
    ~common_bridgespec_dflash_sidecar();

    common_bridgespec_dflash_sidecar(const common_bridgespec_dflash_sidecar &) = delete;
    common_bridgespec_dflash_sidecar & operator=(const common_bridgespec_dflash_sidecar &) = delete;

    bool load(const std::string & library_path,
              const std::string & artifact_dir,
              int32_t encoded_width,
              int32_t block_size,
              std::string & error);

    bool active() const;
    void disable();

    bool get_state(std::vector<uint8_t> & data) const;
    bool set_state(const std::vector<uint8_t> & data) const;
    bool reset_state() const;
    bool truncate_state(int32_t pos_max) const;
    bool rebase_state(int32_t pos_min, int32_t pos_max, int32_t delta) const;

    int chunk(const int32_t * positions, const float * target_features, int count) const;
    int draft(int32_t last_token, int32_t past_tokens, int32_t * output_ids) const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
