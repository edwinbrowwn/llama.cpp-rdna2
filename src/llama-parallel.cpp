#include "llama-parallel.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_set>

bool llama_parallel_topology::hybrid() const {
    return pp_size > 1;
}

bool llama_parallel_topology::tensor_parallel() const {
    return tp_size > 1;
}

bool llama_parallel_topology::pipeline_parallel() const {
    return pp_size > 1;
}

const llama_parallel_group * llama_parallel_topology::group_for_stage(uint32_t stage) const {
    return stage < groups.size() ? &groups[stage] : nullptr;
}

const llama_parallel_group * llama_parallel_topology::group_for_device(ggml_backend_dev_t dev) const {
    for (const auto & group : groups) {
        if (std::find(group.physical_devices.begin(), group.physical_devices.end(), dev) != group.physical_devices.end()) {
            return &group;
        }
    }
    return nullptr;
}

const llama_parallel_group * llama_parallel_topology::group_for_layer(uint32_t il) const {
    for (const auto & group : groups) {
        if (il >= group.layer_begin && il < group.layer_end) {
            return &group;
        }
    }
    return nullptr;
}

static bool normalize_split(
        const float * split,
        size_t count,
        const char * name,
        std::vector<float> & normalized,
        std::string & error) {
    normalized.assign(count, 1.0f / count);
    if (split == nullptr) {
        return true;
    }

    bool all_zero = true;
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const float value = split[i];
        if (!std::isfinite(value) || value < 0.0f) {
            error = std::string(name) + " values must be finite and non-negative";
            return false;
        }
        all_zero = all_zero && value == 0.0f;
        sum += value;
    }
    if (all_zero) {
        return true;
    }
    if (!(sum > 0.0) || !std::isfinite(sum)) {
        error = std::string(name) + " must have a finite positive sum";
        return false;
    }

    for (size_t i = 0; i < count; ++i) {
        if (split[i] <= 0.0f) {
            error = std::string(name) + " must give every rank/stage a positive weight";
            return false;
        }
        normalized[i] = (float) (split[i] / sum);
    }
    return true;
}

bool llama_parallel_topology_build(
        const std::vector<ggml_backend_dev_t> & physical_devices,
        uint32_t tp_size,
        uint32_t pp_size,
        const float * tp_split,
        const float * pp_split,
        llama_parallel_topology & topology,
        std::string & error) {
    topology = {};
    topology.tp_size = tp_size;
    topology.pp_size = pp_size == 0 ? 1 : pp_size;
    error.clear();

    // PP1 deliberately remains on the existing path. Do not infer or rebuild
    // the legacy one-Meta topology here.
    if (topology.pp_size == 1) {
        return true;
    }

    if (tp_size == 0) {
        error = "hybrid parallelism requires --tp-size greater than zero";
        return false;
    }
    if (physical_devices.empty()) {
        error = "hybrid parallelism requires an explicit non-empty device list";
        return false;
    }
    if ((uint64_t) tp_size * topology.pp_size != physical_devices.size()) {
        error = "hybrid parallelism requires physical device count == tp_size * pp_size";
        return false;
    }

    std::unordered_set<ggml_backend_dev_t> unique_devices;
    for (ggml_backend_dev_t dev : physical_devices) {
        if (dev == nullptr) {
            error = "hybrid parallelism device list contains a null device";
            return false;
        }
        if (!unique_devices.insert(dev).second) {
            error = "hybrid parallelism device list contains a duplicate device";
            return false;
        }
    }

    std::vector<float> normalized_tp;
    if (!normalize_split(tp_split, tp_size, "TP split", normalized_tp, error)) {
        return false;
    }
    if (!normalize_split(pp_split, topology.pp_size, "PP split", topology.pp_split, error)) {
        return false;
    }

    topology.groups.reserve(topology.pp_size);
    for (uint32_t stage = 0; stage < topology.pp_size; ++stage) {
        llama_parallel_group group;
        group.stage_index = stage;
        group.tp_split = normalized_tp;
        const auto first = physical_devices.begin() + (size_t) stage * tp_size;
        group.physical_devices.assign(first, first + tp_size);
        topology.groups.push_back(std::move(group));
    }
    return true;
}

bool llama_parallel_topology_assign_layers(
        llama_parallel_topology & topology,
        uint32_t n_layers,
        std::string & error) {
    error.clear();
    if (!topology.hybrid() || topology.groups.size() != topology.pp_size || topology.pp_split.size() != topology.pp_size) {
        error = "layer assignment requires a validated hybrid topology";
        return false;
    }
    if (n_layers < topology.pp_size) {
        error = "hybrid parallelism requires at least one transformer layer per PP stage";
        return false;
    }

    std::vector<float> cumulative(topology.pp_split.size());
    std::partial_sum(topology.pp_split.begin(), topology.pp_split.end(), cumulative.begin());
    cumulative.back() = 1.0f;

    std::vector<uint32_t> layer_counts(topology.pp_size, 0);
    for (uint32_t il = 0; il < n_layers; ++il) {
        const float position = (float) il / n_layers;
        const auto it = std::upper_bound(cumulative.begin(), cumulative.end(), position);
        const size_t stage = std::min<size_t>(it - cumulative.begin(), topology.pp_size - 1);
        layer_counts[stage]++;
    }

    uint32_t begin = 0;
    for (uint32_t stage = 0; stage < topology.pp_size; ++stage) {
        if (layer_counts[stage] == 0) {
            error = "PP split assigns zero transformer layers to a stage";
            return false;
        }
        const uint32_t end = begin + layer_counts[stage];
        topology.groups[stage].layer_begin = begin;
        topology.groups[stage].layer_end   = end;
        begin = end;
    }
    if (begin != n_layers) {
        error = "PP layer assignment does not cover all transformer layers";
        return false;
    }
    return true;
}
