#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpp.h"

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct split_ud {
    size_t ndev;
};

static ggml_backend_meta_split_state split_state_callback(const ggml_tensor * tensor, void * userdata) {
    const auto * ud = static_cast<const split_ud *>(userdata);
    ggml_backend_meta_split_state state{};
    state.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    state.nr[0] = 1;
    state.n_segments = 1;

    if (std::strcmp(tensor->name, "root") == 0 ||
            std::strcmp(tensor->name, "axis3") == 0 ||
            std::strcmp(tensor->name, "axis3-permuted") == 0) {
        const int axis = (std::strcmp(tensor->name, "axis3") == 0 ||
                std::strcmp(tensor->name, "axis3-permuted") == 0) ? 3 : 2;
        if (ud->ndev == 0 || tensor->ne[axis] % (int64_t) ud->ndev != 0) {
            std::fprintf(stderr, "invalid test split: axis=%d ne=%lld ndev=%zu\n", axis, (long long) tensor->ne[axis], ud->ndev);
            std::abort();
        }
        state.axis = axis == 3 ? GGML_BACKEND_SPLIT_AXIS_3 : GGML_BACKEND_SPLIT_AXIS_2;
        for (size_t j = 0; j < ud->ndev; ++j) {
            state.ne[j] = tensor->ne[axis] / (int64_t) ud->ndev;
        }
    } else if (std::strcmp(tensor->name, "partial") == 0) {
        state.axis = GGML_BACKEND_SPLIT_AXIS_PARTIAL;
    } else if (std::strcmp(tensor->name, "segments") == 0) {
        state.axis = GGML_BACKEND_SPLIT_AXIS_0;
        state.n_segments = 2;
        state.nr[0] = 1;
        state.nr[1] = 1;
        for (size_t j = 0; j < ud->ndev; ++j) {
            state.ne[j] = tensor->ne[0] / 2 / (int64_t) ud->ndev;
            state.ne[ud->ndev + j] = tensor->ne[0] / 2 / (int64_t) ud->ndev;
        }
    }
    return state;
}

struct fake_event_device_context {
    const char * name;
    bool fail_event_new;
    int created = 0;
    int freed   = 0;
};

static const char * fake_event_device_name(ggml_backend_dev_t dev) {
    return ((fake_event_device_context *) dev->context)->name;
}

static const char * fake_event_device_description(ggml_backend_dev_t dev) {
    return fake_event_device_name(dev);
}

static void fake_event_device_memory(ggml_backend_dev_t, size_t * free, size_t * total) {
    *free = 0;
    *total = 0;
}

static enum ggml_backend_dev_type fake_event_device_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void fake_event_device_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    auto * ctx = (fake_event_device_context *) dev->context;
    *props = {};
    props->name        = ctx->name;
    props->description = ctx->name;
    props->type        = GGML_BACKEND_DEVICE_TYPE_GPU;
    props->caps.async  = true;
    props->caps.events = true;
}

static bool fake_event_device_supports_op(ggml_backend_dev_t, const ggml_tensor *) {
    return false;
}

static bool fake_event_device_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t) {
    return false;
}

static ggml_backend_event_t fake_event_new(ggml_backend_dev_t dev) {
    auto * ctx = (fake_event_device_context *) dev->context;
    ctx->created++;
    if (ctx->fail_event_new) {
        return nullptr;
    }
    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ nullptr,
    };
}

static void fake_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ((fake_event_device_context *) dev->context)->freed++;
    delete event;
}

static void fake_event_synchronize(ggml_backend_dev_t, ggml_backend_event_t) {
}

static const ggml_backend_device_i fake_event_device_iface = {
    /* .get_name             = */ fake_event_device_name,
    /* .get_description      = */ fake_event_device_description,
    /* .get_memory           = */ fake_event_device_memory,
    /* .get_type             = */ fake_event_device_type,
    /* .get_props            = */ fake_event_device_props,
    /* .init_backend         = */ nullptr,
    /* .get_buffer_type      = */ nullptr,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ fake_event_device_supports_op,
    /* .supports_buft        = */ fake_event_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ fake_event_new,
    /* .event_free           = */ fake_event_free,
    /* .event_synchronize    = */ fake_event_synchronize,
};

static bool test_meta_events(ggml_backend_dev_t meta_dev, ggml_backend_t backend) {
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(meta_dev, &props);
    if (!props.caps.events) {
        std::fprintf(stderr, "Meta device did not advertise child-composite events\n");
        return false;
    }

    std::vector<ggml_backend_event_t> events;
    for (size_t i = 0; i < 4; ++i) {
        ggml_backend_event_t event = ggml_backend_event_new(meta_dev);
        if (event == nullptr) {
            std::fprintf(stderr, "failed to create Meta event %zu\n", i);
            for (ggml_backend_event_t allocated : events) {
                ggml_backend_event_free(allocated);
            }
            return false;
        }
        events.push_back(event);
    }
    for (ggml_backend_event_t event : events) {
        ggml_backend_event_record(event, backend);
        ggml_backend_event_wait(backend, event);
    }
    for (ggml_backend_event_t event : events) {
        ggml_backend_event_synchronize(event);
        ggml_backend_event_free(event);
    }

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_ptr cpu_backend(cpu_dev ? ggml_backend_dev_init(cpu_dev, nullptr) : nullptr);
    if (!cpu_backend) {
        std::fprintf(stderr, "failed to initialize CPU backend for scheduler event-slot test\n");
        return false;
    }
    ggml_backend_t sched_backends[] = {backend, cpu_backend.get()};
    ggml_backend_buffer_type_t sched_bufts[] = {
        ggml_backend_get_default_buffer_type(backend),
        ggml_backend_get_default_buffer_type(cpu_backend.get()),
    };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(
            sched_backends, sched_bufts, 2, 1024, /*parallel =*/ true, /*op_offload =*/ false));
    if (!sched || ggml_backend_sched_get_n_copies(sched.get()) != 4) {
        std::fprintf(stderr, "Meta scheduler did not create four event-protected copy slots\n");
        return false;
    }
    return true;
}

static bool test_meta_event_partial_cleanup() {
    fake_event_device_context ok_ctx   = {"FakeEventOK",   false};
    fake_event_device_context fail_ctx = {"FakeEventFail", true};
    ggml_backend_device ok_dev   = {fake_event_device_iface, nullptr, &ok_ctx};
    ggml_backend_device fail_dev = {fake_event_device_iface, nullptr, &fail_ctx};
    ggml_backend_dev_t children[] = {&ok_dev, &fail_dev};
    split_ud ud{2};
    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(children, 2, split_state_callback, &ud);
    ggml_backend_event_t event = ggml_backend_event_new(meta_dev);
    if (event != nullptr) {
        std::fprintf(stderr, "Meta event creation unexpectedly survived a child allocation failure\n");
        ggml_backend_event_free(event);
        return false;
    }
    if (ok_ctx.created != 1 || ok_ctx.freed != 1 || fail_ctx.created != 1 || fail_ctx.freed != 0) {
        std::fprintf(stderr, "partial Meta event cleanup mismatch: ok=%d/%d fail=%d/%d\n",
                ok_ctx.created, ok_ctx.freed, fail_ctx.created, fail_ctx.freed);
        return false;
    }
    return true;
}

static bool test_deep_meta_graph(ggml_backend_t backend) {
    static constexpr size_t depth = 2048;
    const ggml_init_params params = {
        /*.mem_size   =*/ 32*1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        std::fprintf(stderr, "failed to initialize deep-graph context\n");
        return false;
    }

    ggml_tensor * root = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
    ggml_set_name(root, "deep-root");
    std::vector<ggml_tensor *> chain = {root};
    chain.reserve(depth + 1);
    ggml_tensor * output = root;
    for (size_t i = 0; i < depth; ++i) {
        output = ggml_scale(ctx.get(), output, 1.0f);
        chain.push_back(output);
    }
    ggml_set_name(output, "deep-output");

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), depth + 1, false);
    ggml_build_forward_expand(graph, output);
    if ((size_t) ggml_graph_n_nodes(graph) != depth) {
        std::fprintf(stderr, "deep graph has %d nodes, expected %zu\n", ggml_graph_n_nodes(graph), depth);
        return false;
    }

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buffer) {
        std::fprintf(stderr, "failed to allocate deep-graph tensors\n");
        return false;
    }
    ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    // Invalidate allocation-time identity snapshots to model compute-arena
    // address reuse. Resolving the output must rebuild the entire dependency
    // chain without recursing once per node.
    for (size_t i = 0; i < chain.size(); ++i) {
        ggml_format_name(chain[i], "deep-mutated-%zu", i);
    }

    const float expected = 0.125f;
    ggml_backend_tensor_set(root, &expected, 0, sizeof(expected));

    // Force graph-external shard resolution from the deepest output before
    // normal topological graph execution. Recursive resolution needs more than
    // the default process stack for a chain of this depth.
    float unresolved = 0.0f;
    ggml_backend_tensor_get(output, &unresolved, 0, sizeof(unresolved));

    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "deep meta graph compute failed: %s\n", ggml_status_to_string(status));
        return false;
    }
    ggml_backend_synchronize(backend);

    float actual = 0.0f;
    ggml_backend_tensor_get(output, &actual, 0, sizeof(actual));
    if (actual != expected) {
        std::fprintf(stderr, "deep meta graph mismatch: %.9g != %.9g\n", actual, expected);
        return false;
    }

    std::printf("deep meta graph passed (%zu nodes)\n", depth);
    return true;
}

int main() {
    ggml_backend_load_all();

    std::vector<ggml_backend_dev_t> simple_devs;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_buffer_type(dev) != ggml_backend_cpu_buffer_type()) {
            simple_devs.push_back(dev);
        }
    }
    if (simple_devs.size() < 2) {
        std::puts("meta split readback test skipped: fewer than two non-CPU devices");
        return 0;
    }

    split_ud ud{simple_devs.size()};
    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(simple_devs.data(), simple_devs.size(), split_state_callback, &ud);
    const std::string meta_name = ggml_backend_dev_name(meta_dev);
    for (ggml_backend_dev_t simple_dev : simple_devs) {
        if (meta_name.find(ggml_backend_dev_name(simple_dev)) == std::string::npos) {
            std::fprintf(stderr, "Meta device name '%s' omits child '%s'\n", meta_name.c_str(), ggml_backend_dev_name(simple_dev));
            return 1;
        }
    }
    ggml_backend_ptr backend(ggml_backend_dev_init(meta_dev, nullptr));
    if (!backend) {
        std::fprintf(stderr, "failed to initialize meta backend\n");
        return 1;
    }
    if (!test_meta_events(meta_dev, backend.get())) {
        return 1;
    }

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_dev != nullptr) {
        ggml_backend_dev_t mixed_children[] = {simple_devs.front(), cpu_dev};
        split_ud mixed_ud{2};
        ggml_backend_dev_t mixed_meta = ggml_backend_meta_device(mixed_children, 2, split_state_callback, &mixed_ud);
        ggml_backend_dev_props mixed_props;
        ggml_backend_dev_get_props(mixed_meta, &mixed_props);
        ggml_backend_event_t mixed_event = ggml_backend_event_new(mixed_meta);
        if (mixed_props.caps.events || mixed_event != nullptr) {
            std::fprintf(stderr, "Meta device advertised events with an unsupported child\n");
            ggml_backend_event_free(mixed_event);
            return 1;
        }
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 16*1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        std::fprintf(stderr, "failed to initialize ggml context\n");
        return 1;
    }

    ggml_tensor * root = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 4, 4, 8, 1);
    ggml_set_name(root, "root");
    ggml_tensor * mirror = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 4, 4, 8, 1);
    ggml_set_name(mirror, "mirror");
    ggml_tensor * axis3 = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 3, 4, 4, 8);
    ggml_set_name(axis3, "axis3");
    // Keep the split on axis 3 while permuting dimensions inside each
    // physical row.  This is non-contiguous metadata but each shard remains
    // safe for the axis-3 row-wise transfer path.
    ggml_tensor * axis3_permuted = ggml_permute(ctx.get(), axis3, 1, 0, 2, 3);
    ggml_set_name(axis3_permuted, "axis3-permuted");
    ggml_tensor * partial = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 4, 4, 4, 1);
    ggml_set_name(partial, "partial");
    ggml_tensor * segments = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 8, 4, 1, 1);
    ggml_set_name(segments, "segments");
    // Swap dimensions 0 and 1 while preserving dimension 2.  The result is
    // deliberately non-contiguous but remains split along axis 2.
    ggml_tensor * permuted = ggml_permute(ctx.get(), root, 1, 0, 2, 3);
    ggml_set_name(permuted, "root-permuted");

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!buffer) {
        std::fprintf(stderr, "failed to allocate meta tensors\n");
        return 1;
    }
    const std::string meta_buft_name = ggml_backend_buffer_name(buffer.get());
    for (ggml_backend_dev_t simple_dev : simple_devs) {
        const char * child_buft_name = ggml_backend_buft_name(ggml_backend_dev_buffer_type(simple_dev));
        if (meta_buft_name.find(child_buft_name) == std::string::npos) {
            std::fprintf(stderr, "Meta buffer name '%s' omits child buffer '%s'\n", meta_buft_name.c_str(), child_buft_name);
            return 1;
        }
    }

    const size_t nbytes = ggml_nbytes(root);
    std::vector<float> expected(nbytes / sizeof(float));
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = std::sin((float) i * 0.125f);
    }
    ggml_backend_tensor_set(root, expected.data(), 0, nbytes);

    std::vector<float> actual(expected.size(), 0.0f);
    ggml_backend_tensor_get(root, actual.data(), 0, nbytes);
    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != actual[i]) {
            std::fprintf(stderr, "contiguous axis-2 readback mismatch at %zu: %.9g != %.9g\n", i, expected[i], actual[i]);
            return 1;
        }
    }

    std::fill(actual.begin(), actual.end(), 0.0f);
    ggml_backend_tensor_get(permuted, actual.data(), 0, nbytes);
    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != actual[i]) {
            std::fprintf(stderr, "permuted axis-2 readback mismatch at %zu: %.9g != %.9g\n", i, expected[i], actual[i]);
            return 1;
        }
    }

    std::vector<float> mirrored(expected.size());
    for (size_t i = 0; i < mirrored.size(); ++i) {
        mirrored[i] = std::cos((float) i * 0.0625f);
    }
    ggml_backend_tensor_set(mirror, mirrored.data(), 0, nbytes);
    std::fill(actual.begin(), actual.end(), 0.0f);
    ggml_backend_tensor_get(mirror, actual.data(), 0, nbytes);
    for (size_t i = 0; i < mirrored.size(); ++i) {
        if (mirrored[i] != actual[i]) {
            std::fprintf(stderr, "mirrored readback mismatch at %zu: %.9g != %.9g\n", i, mirrored[i], actual[i]);
            return 1;
        }
    }

    const size_t axis3_nbytes = ggml_nbytes(axis3);
    std::vector<float> axis3_expected(axis3_nbytes / sizeof(float));
    for (size_t i = 0; i < axis3_expected.size(); ++i) {
        axis3_expected[i] = (float) (i * 3 + 1);
    }
    ggml_backend_tensor_set(axis3, axis3_expected.data(), 0, axis3_nbytes);
    std::vector<float> axis3_actual(axis3_expected.size(), 0.0f);
    ggml_backend_tensor_get(axis3, axis3_actual.data(), 0, axis3_nbytes);
    for (size_t i = 0; i < axis3_expected.size(); ++i) {
        if (axis3_expected[i] != axis3_actual[i]) {
            std::fprintf(stderr, "axis-3 readback mismatch at %zu: %.9g != %.9g\n", i, axis3_expected[i], axis3_actual[i]);
            return 1;
        }
    }

    std::fill(axis3_actual.begin(), axis3_actual.end(), 0.0f);
    ggml_backend_tensor_get(axis3_permuted, axis3_actual.data(), 0, axis3_nbytes);
    if (axis3_actual != axis3_expected) {
        std::fprintf(stderr, "permuted axis-3 readback mismatch\n");
        return 1;
    }

    const size_t axis3_row_bytes = axis3->nb[3];
    const size_t axis3_row_elems = axis3_row_bytes / sizeof(float);
    const size_t patch_row_start = 2;
    const size_t patch_row_count = 3;
    std::vector<float> patch(patch_row_count * axis3_row_elems, -7.0f);
    ggml_backend_tensor_set(axis3, patch.data(), patch_row_start * axis3_row_bytes, patch.size() * sizeof(float));
    std::vector<float> patched_expected = axis3_expected;
    std::copy(patch.begin(), patch.end(), patched_expected.begin() + patch_row_start * axis3_row_elems);
    std::fill(axis3_actual.begin(), axis3_actual.end(), 0.0f);
    ggml_backend_tensor_get(axis3, axis3_actual.data(), 0, axis3_nbytes);
    if (axis3_actual != patched_expected) {
        std::fprintf(stderr, "axis-3 partial set/readback mismatch\n");
        return 1;
    }

    std::fill(axis3_actual.begin(), axis3_actual.end(), 0.0f);
    ggml_backend_tensor_set_async(backend.get(), axis3, axis3_expected.data(), 0, axis3_nbytes);
    ggml_backend_synchronize(backend.get());
    ggml_backend_tensor_get_async(backend.get(), axis3, axis3_actual.data(), 0, axis3_nbytes);
    ggml_backend_synchronize(backend.get());
    for (size_t i = 0; i < axis3_expected.size(); ++i) {
        if (axis3_expected[i] != axis3_actual[i]) {
            std::fprintf(stderr, "axis-3 async readback mismatch at %zu: %.9g != %.9g\n", i, axis3_expected[i], axis3_actual[i]);
            return 1;
        }
    }

    const size_t partial_nbytes = ggml_nbytes(partial);
    std::vector<float> partial_expected(partial_nbytes / sizeof(float));
    for (size_t i = 0; i < partial_expected.size(); ++i) {
        partial_expected[i] = (float) (i + 0.25f);
    }
    ggml_backend_tensor_set(partial, partial_expected.data(), 0, partial_nbytes);
    std::vector<float> partial_actual(partial_expected.size(), 0.0f);
    ggml_backend_tensor_get(partial, partial_actual.data(), 0, partial_nbytes);
    if (partial_actual != partial_expected) {
        std::fprintf(stderr, "partial set/readback mismatch\n");
        return 1;
    }
    std::fill(partial_actual.begin(), partial_actual.end(), 0.0f);
    ggml_backend_tensor_set_async(backend.get(), partial, partial_expected.data(), 0, partial_nbytes);
    ggml_backend_synchronize(backend.get());
    ggml_backend_tensor_get_async(backend.get(), partial, partial_actual.data(), 0, partial_nbytes);
    ggml_backend_synchronize(backend.get());
    if (partial_actual != partial_expected) {
        std::fprintf(stderr, "partial async set/readback mismatch\n");
        return 1;
    }

    const size_t segments_nbytes = ggml_nbytes(segments);
    std::vector<float> segments_expected(segments_nbytes / sizeof(float));
    for (size_t i = 0; i < segments_expected.size(); ++i) {
        segments_expected[i] = (float) (1000 + i);
    }
    ggml_backend_tensor_set(segments, segments_expected.data(), 0, segments_nbytes);
    std::vector<float> segments_actual(segments_expected.size(), 0.0f);
    ggml_backend_tensor_get(segments, segments_actual.data(), 0, segments_nbytes);
    if (segments_actual != segments_expected) {
        std::fprintf(stderr, "multi-segment readback mismatch\n");
        return 1;
    }

    if (!test_deep_meta_graph(backend.get())) {
        return 1;
    }
    if (!test_meta_event_partial_cleanup()) {
        return 1;
    }

    std::puts("meta split/event axis-2, axis-3, mirrored, partial, and multi-segment tests passed");
    return 0;
}
