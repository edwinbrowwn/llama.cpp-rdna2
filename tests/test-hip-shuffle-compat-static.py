#!/usr/bin/env python3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    hip = read("ggml/src/ggml-cuda/vendors/hip.h")

    disable_pos = hip.find("#define HIP_DISABLE_WARP_SYNC_BUILTINS 1")
    runtime_pos = hip.find("#include <hip/hip_runtime.h>")
    require(disable_pos >= 0, "HIP sync warp builtins must be disabled for ROCm compatibility")
    require(runtime_pos >= 0, "HIP runtime include not found")
    require(disable_pos < runtime_pos, "HIP_DISABLE_WARP_SYNC_BUILTINS must be defined before hip_runtime.h")

    required_helpers = [
        "GGML_HIP_SHFL_SYNC_3",
        "GGML_HIP_SHFL_SYNC_4",
        "GGML_HIP_SHFL_UP_SYNC_3",
        "GGML_HIP_SHFL_UP_SYNC_4",
        "GGML_HIP_SHFL_DOWN_SYNC_3",
        "GGML_HIP_SHFL_DOWN_SYNC_4",
        "GGML_HIP_SHFL_XOR_SYNC_3",
        "GGML_HIP_SHFL_XOR_SYNC_4",
    ]
    for helper in required_helpers:
        require(helper in hip, f"missing HIP shuffle helper {helper}")

    required_shims = [
        "__shfl_sync(...)",
        "__shfl_up_sync(...)",
        "__shfl_down_sync(...)",
        "__shfl_xor_sync(...)",
    ]
    for shim in required_shims:
        require(shim in hip, f"missing variadic HIP compatibility shim for {shim}")

    require("__shfl(var, srcLane)" in hip, "3-argument __shfl_sync must map to legacy HIP default width")
    require("__shfl(var, srcLane, width)" in hip, "4-argument __shfl_sync must preserve explicit width")
    require("__shfl_up(var, delta)" in hip, "3-argument __shfl_up_sync must map to legacy HIP default width")
    require("__shfl_up(var, delta, width)" in hip, "4-argument __shfl_up_sync must preserve explicit width")
    require("__shfl_down(var, delta)" in hip, "3-argument __shfl_down_sync must map to legacy HIP default width")
    require("__shfl_down(var, delta, width)" in hip, "4-argument __shfl_down_sync must preserve explicit width")
    require("__shfl_xor(var, laneMask)" in hip, "3-argument __shfl_xor_sync must map to legacy HIP default width")
    require("__shfl_xor(var, laneMask, width)" in hip, "4-argument __shfl_xor_sync must preserve explicit width")


if __name__ == "__main__":
    main()
