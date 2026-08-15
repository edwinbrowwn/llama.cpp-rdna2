#include "llama-parallel.h"
#include "llama.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#undef NDEBUG
#include <cassert>

static ggml_backend_dev_t fake_dev(uintptr_t id) {
    return reinterpret_cast<ggml_backend_dev_t>(id);
}

static bool near(float a, float b) {
    return std::fabs(a - b) < 1e-6f;
}

int main() {
    const llama_model_params defaults = llama_model_default_params();
    assert(defaults.tp_size == 0);
    assert(defaults.pp_size == 1);
    assert(defaults.pp_split == nullptr);

    const std::vector<ggml_backend_dev_t> devs = {
        fake_dev(1), fake_dev(2), fake_dev(3), fake_dev(4),
    };

    llama_parallel_topology topology;
    std::string error;

    // PP1 is deliberately a legacy no-op even when a TP size is supplied.
    assert(llama_parallel_topology_build(devs, 4, 1, nullptr, nullptr, topology, error));
    assert(!topology.hybrid());
    assert(topology.groups.empty());

    const float tp_equal[] = {1.0f, 1.0f};
    const float pp_equal[] = {1.0f, 1.0f};
    assert(llama_parallel_topology_build(devs, 2, 2, tp_equal, pp_equal, topology, error));
    assert(topology.hybrid());
    assert(topology.tensor_parallel());
    assert(topology.pipeline_parallel());
    assert(topology.groups.size() == 2);
    assert(topology.groups[0].physical_devices == std::vector<ggml_backend_dev_t>({devs[0], devs[1]}));
    assert(topology.groups[1].physical_devices == std::vector<ggml_backend_dev_t>({devs[2], devs[3]}));
    assert(near(topology.groups[0].tp_split[0], 0.5f));
    assert(near(topology.groups[1].tp_split[1], 0.5f));
    assert(topology.group_for_stage(0) == &topology.groups[0]);
    assert(topology.group_for_device(devs[3]) == &topology.groups[1]);
    assert(topology.group_for_device(fake_dev(9)) == nullptr);

    assert(llama_parallel_topology_assign_layers(topology, 40, error));
    assert(topology.groups[0].layer_begin == 0);
    assert(topology.groups[0].layer_end == 20);
    assert(topology.groups[1].layer_begin == 20);
    assert(topology.groups[1].layer_end == 40);
    assert(topology.group_for_layer(19) == &topology.groups[0]);
    assert(topology.group_for_layer(20) == &topology.groups[1]);
    assert(topology.group_for_layer(40) == nullptr);

    // Device order defines physical grouping.
    const std::vector<ggml_backend_dev_t> reordered = {
        devs[0], devs[2], devs[1], devs[3],
    };
    assert(llama_parallel_topology_build(reordered, 2, 2, nullptr, nullptr, topology, error));
    assert(topology.groups[0].physical_devices[0] == devs[0]);
    assert(topology.groups[0].physical_devices[1] == devs[2]);
    assert(topology.groups[1].physical_devices[0] == devs[1]);
    assert(topology.groups[1].physical_devices[1] == devs[3]);

    // Non-equal PP weights produce the requested neighboring boundary.
    const float pp_18_22[] = {18.0f, 22.0f};
    assert(llama_parallel_topology_build(devs, 2, 2, tp_equal, pp_18_22, topology, error));
    assert(llama_parallel_topology_assign_layers(topology, 40, error));
    assert(topology.groups[0].layer_begin == 0);
    assert(topology.groups[0].layer_end == 18);
    assert(topology.groups[1].layer_begin == 18);
    assert(topology.groups[1].layer_end == 40);

    // All-zero splits retain the established equal/default convention.
    const float split_zero[] = {0.0f, 0.0f};
    assert(llama_parallel_topology_build(devs, 2, 2, split_zero, split_zero, topology, error));
    assert(near(topology.groups[0].tp_split[0], 0.5f));
    assert(near(topology.pp_split[1], 0.5f));

    // Invalid dimensions, duplicates, weights, and zero-layer stages fail clearly.
    assert(!llama_parallel_topology_build(devs, 3, 2, nullptr, nullptr, topology, error));
    assert(error.find("device count") != std::string::npos);
    assert(!llama_parallel_topology_build(devs, 0, 2, nullptr, nullptr, topology, error));
    assert(error.find("tp-size") != std::string::npos);

    auto duplicate = devs;
    duplicate[3] = duplicate[0];
    assert(!llama_parallel_topology_build(duplicate, 2, 2, nullptr, nullptr, topology, error));
    assert(error.find("duplicate") != std::string::npos);

    const float negative[] = {1.0f, -1.0f};
    assert(!llama_parallel_topology_build(devs, 2, 2, negative, nullptr, topology, error));
    assert(error.find("finite and non-negative") != std::string::npos);

    const float zero_stage[] = {1.0f, 0.0f};
    assert(!llama_parallel_topology_build(devs, 2, 2, nullptr, zero_stage, topology, error));
    assert(error.find("every rank/stage") != std::string::npos);

    assert(llama_parallel_topology_build(devs, 2, 2, nullptr, nullptr, topology, error));
    assert(!llama_parallel_topology_assign_layers(topology, 1, error));
    assert(error.find("at least one transformer layer") != std::string::npos);

    return 0;
}
