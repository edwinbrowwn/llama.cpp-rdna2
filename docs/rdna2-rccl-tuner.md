# Native RDNA2 RCCL tuning

When built with `GGML_HIP=ON` and `GGML_HIP_RCCL=ON`, `llama-server` can select the validated V620 RCCL policy before backend communicator and HIP-graph initialization.

```bash
llama-server --rccl-tune auto \
  -m Qwen3.8-27B-Q4_0.gguf \
  -sm tensor -dev ROCm0,ROCm1,ROCm2,ROCm3 -ts 1,1,1,1 \
  -ngl 999
```

`auto` is the default for `llama-server`. The native C++ startup gate requires tensor-parallel mode, a 2- to 8-way nonzero tensor split, RCCL, and a plugin built beside the executable. The RCCL v6 plugin then requires one node with 2 to 8 visible AMD Radeon Pro V620 (`gfx1030`) devices. It applies Ring/LL with `min(3, TP-ranks)` channels only to the exact 20,480-byte AllReduce. All other collectives remain RCCL Auto.

TP4 is the measured/certified default policy. TP2 through TP8 are experimental and require explicit `--rccl-tune force`; they are not benchmarked at startup. This makes the broader policy opt-in while retaining conservative Auto behavior for unvalidated TP sizes.

Modes:

- `auto`: enable the offline-certified TP4 policy when the strict launch gate passes; TP2/3/5/6/7/8 remain Auto.
- `force`: explicitly enable the experimental policy for supported TP2-TP8 shapes, still subject to plugin eligibility and conflict checks.
- `off`: do not select the native policy.

Equivalent environment variable: `LLAMA_ARG_RCCL_TUNE=auto|force|off`. Existing `NCCL_ALGO`, `NCCL_PROTO`, channel, thread, or tuner-plugin overrides disable automatic selection rather than being overwritten.

The plugin is built as part of the llama.cpp build and emitted beside `llama-server` as `libnccl-tuner-rdna2-v620.so`; Python and the external launcher are not required for native server startup.

Example build:

```bash
cmake -S . -B build -G Ninja \
  -DGGML_HIP=ON -DGGML_HIP_RCCL=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1030 \
  -DCMAKE_PREFIX_PATH=/opt/rocm/core-7.14
cmake --build build --target llama-server ggml-rccl-tuner-rdna2-v620
```

Startup logs identify whether native auto selected the policy, skipped because of shape/conflicts, or left RCCL Auto in control.
