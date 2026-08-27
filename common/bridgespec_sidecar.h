#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Host-side loader for the optional BridgeSpec Qwen3.8-27B sidecars.
// The sidecars are deliberately opt-in and model-specific. A loader object
// keeps the library resident for the lifetime of the process because the
// current release ABI has no shutdown operation. State/KV calls are serialized
// by the speculative driver and sequence-scoped in ABI 3/4.
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
              int32_t n_seq,
              std::string & error);

    bool active() const;
    void disable();

    bool get_state(int32_t seq_id, std::vector<uint8_t> & data) const;
    bool set_state(int32_t seq_id, const std::vector<uint8_t> & data) const;
    bool reset_state(int32_t seq_id) const;
    bool truncate_state(int32_t seq_id, int32_t pos_max) const;
    bool commit_state(int32_t seq_id, int32_t pos_max, const float * hidden_device) const;
    bool rebase_state(int32_t seq_id, int32_t pos_min, int32_t pos_max, int32_t delta) const;
    bool attach_target_stream(void * stream, int32_t device) const;

    int catchup(int32_t seq_id, const int32_t * tokens, const int32_t * positions,
                const float * hidden_rows, int count) const;
    int catchup_device(int32_t seq_id, const int32_t * tokens, const int32_t * positions,
                       const float * hidden_rows_device, int count) const;
    int draft(int32_t seq_id, int32_t last_token, int32_t past_tokens,
              const float * hidden, int max_draft, int32_t * output_ids) const;
    int draft_device(int32_t seq_id, int32_t last_token, int32_t past_tokens,
                     int max_draft, int32_t * output_ids) const;

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
              int32_t n_seq,
              std::string & error);

    bool active() const;
    void disable();

    bool get_state(int32_t seq_id, std::vector<uint8_t> & data) const;
    bool set_state(int32_t seq_id, const std::vector<uint8_t> & data) const;
    bool reset_state(int32_t seq_id) const;
    bool truncate_state(int32_t seq_id, int32_t pos_max) const;
    bool commit_state(int32_t seq_id, int32_t pos_max) const;
    bool rebase_state(int32_t seq_id, int32_t pos_min, int32_t pos_max, int32_t delta) const;
    bool attach_target_stream(void * stream, int32_t device) const;

    int chunk(int32_t seq_id, const int32_t * positions,
              const float * target_features, int count) const;
    int chunk_device(int32_t seq_id, const int32_t * positions,
                     const void * const * target_layer_features_device,
                     int n_layers, int layer_width, int count) const;
    int draft(int32_t seq_id, int32_t last_token, int32_t past_tokens,
              int32_t * output_ids) const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
