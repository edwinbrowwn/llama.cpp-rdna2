#pragma once

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Internal, experimental description of explicit tensor-parallel groups arranged
// as pipeline stages. Legacy PP1 execution does not consume this topology.
struct llama_parallel_group {
    uint32_t stage_index = 0;

    std::vector<ggml_backend_dev_t> physical_devices;
    std::vector<float> tp_split;

    // Logical Meta device created from physical_devices. Populated only by
    // explicit hybrid model initialization; pure topology construction leaves
    // it null.
    ggml_backend_dev_t meta_device = nullptr;

    // Half-open transformer-layer range. Populated by
    // llama_parallel_topology_assign_layers().
    uint32_t layer_begin = 0;
    uint32_t layer_end   = 0;
};

struct llama_parallel_topology {
    uint32_t tp_size = 0;
    uint32_t pp_size = 1;

    // Normalized stage weights, one per PP stage in hybrid mode.
    std::vector<float> pp_split;
    std::vector<llama_parallel_group> groups;

    bool hybrid() const;
    bool tensor_parallel() const;
    bool pipeline_parallel() const;

    const llama_parallel_group * group_for_stage(uint32_t stage) const;
    const llama_parallel_group * group_for_device(ggml_backend_dev_t dev) const;
    const llama_parallel_group * group_for_meta_device(ggml_backend_dev_t dev) const;
    const llama_parallel_group * group_for_layer(uint32_t il) const;
};

// Build and validate an explicit hybrid topology. pp_size <= 1 describes the
// legacy path and intentionally leaves groups empty. For pp_size > 1, devices
// are partitioned into consecutive, uniform TP groups.
bool llama_parallel_topology_build(
        const std::vector<ggml_backend_dev_t> & physical_devices,
        uint32_t tp_size,
        uint32_t pp_size,
        const float * tp_split,
        const float * pp_split,
        llama_parallel_topology & topology,
        std::string & error);

// Assign complete transformer layers to stages using the normalized PP weights.
// Every stage must receive at least one layer.
bool llama_parallel_topology_assign_layers(
        llama_parallel_topology & topology,
        uint32_t n_layers,
        std::string & error);
