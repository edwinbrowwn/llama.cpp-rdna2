// SPDX-License-Identifier: MIT
#include "artifact_manifest.h"
#include "bridgespec_sidecar.h"
#include "../include/bridgespec/sidecar_abi.h"

#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

using bridgespec_artifact::ManifestParser;
using bridgespec_artifact::TensorDesc;

static bool parses(const char * json, std::vector<TensorDesc> & tensors) {
    const std::string text(json);
    std::string error;
    ManifestParser parser(text, error);
    return parser.parse(tensors);
}

static int require(bool condition, const char * label) {
    if (condition) return 0;
    std::fprintf(stderr, "FAILED: %s\n", label);
    return 1;
}

int main() {
    int failures = 0;
    failures += require(sizeof(bridgespec_sidecar_state) == 24,
                        "sidecar state ABI is a fixed 24-byte record");
    failures += require(BRIDGESPEC_STATE_VERSION == 1 &&
                        BRIDGESPEC_STATE_KIND_MTP == 1 && BRIDGESPEC_STATE_KIND_DFLASH == 2,
                        "sidecar state ABI constants are stable");
    failures += require(offsetof(bridgespec_sidecar_state, pos_min) == 8 &&
                        offsetof(bridgespec_sidecar_state, pos_max) == 12 &&
                        offsetof(bridgespec_sidecar_state, epoch) == 16,
                        "sidecar state ABI field offsets are stable");

    std::vector<TensorDesc> tensors;
    const char * valid =
        "{\"schema\":1,\"generator\":{\"name\":\"ignored\"},\"tensors\":["
        "{\"name\":\"a\",\"dtype\":\"0\",\"shape\":[2],\"offset\":0,\"nbytes\":8}]}";
    failures += require(parses(valid, tensors), "valid manifest parses");
    std::string error;
    failures += require(tensors.size() == 1 &&
                        bridgespec_artifact::validate_blob_layout(tensors, 8, error),
                        "valid contiguous layout passes");

    tensors.clear();
    failures += require(parses("{\"tensors\":[]}", tensors),
                        "legacy manifest without schema remains supported");

    tensors.clear();
    failures += require(!parses(
        "{\"schema\":1,\"tensors\":[{\"name\":\"a\",\"dtype\":\"0\",\"shape\":[2],\"offset\":0}]}",
        tensors), "missing required tensor field is rejected");

    tensors.clear();
    failures += require(!parses(
        "{\"schema\":1.0,\"tensors\":[]}", tensors),
        "fractional schema is rejected");

    tensors.clear();
    failures += require(!parses(
        "{\"schema\":2,\"tensors\":[]}", tensors),
        "unsupported schema is rejected");

    tensors.clear();
    failures += require(!parses(
        "{\"schema\":1,\"schema\":1,\"tensors\":[]}", tensors),
        "duplicate schema field is rejected");

    tensors.clear();
    failures += require(!parses(
        "{\"schema\":1,\"tensors\":[{\"name\":\"a\",\"dtype\":\"0\",\"shape\":[2],"
        "\"offset\":18446744073709551616,\"nbytes\":8}]}", tensors),
        "overflowing integer is rejected");

    tensors.clear();
    failures += require(parses(
        "{\"schema\":1,\"metadata\":{\"name\":\"not-a-tensor\",\"offset\":999},\"tensors\":[]}",
        tensors) && tensors.empty(), "nested metadata is not scraped as a tensor");

    tensors.clear();
    failures += require(parses(
        "{\"schema\":1,\"tensors\":["
        "{\"name\":\"a\",\"dtype\":\"0\",\"shape\":[1],\"offset\":0,\"nbytes\":4},"
        "{\"name\":\"a\",\"dtype\":\"0\",\"shape\":[1],\"offset\":4,\"nbytes\":4}]}",
        tensors), "duplicate-name fixture parses structurally");
    error.clear();
    failures += require(!bridgespec_artifact::validate_blob_layout(tensors, 8, error),
                        "duplicate tensor name is rejected by layout validation");

    tensors.clear();
    failures += require(parses(
        "{\"schema\":1,\"tensors\":["
        "{\"name\":\"a\",\"dtype\":\"0\",\"shape\":[1],\"offset\":4,\"nbytes\":4}]}",
        tensors), "gapped-layout fixture parses structurally");
    error.clear();
    failures += require(!bridgespec_artifact::validate_blob_layout(tensors, 8, error),
                        "gapped tensor range is rejected");

    error.clear();
    failures += require(bridgespec_artifact::validate_remap({0, 3, 2}, 4, error),
                        "unique in-range remap passes");
    error.clear();
    failures += require(!bridgespec_artifact::validate_remap({0, 3, 3}, 4, error),
                        "duplicate remap id is rejected");
    error.clear();
    failures += require(!bridgespec_artifact::validate_remap({0, 4}, 4, error),
                        "out-of-range remap id is rejected");
    error.clear();
    failures += require(!bridgespec_artifact::validate_remap({0, -1}, 4, error),
                        "negative remap id is rejected");

    common_bridgespec_mtp_sidecar mtp;
    error.clear();
    failures += require(!mtp.load("relative-sidecar.so", "/absolute/artifacts", "/absolute/ids.bin", 5120, 40960, 1, error) &&
                        error.find("absolute path") != std::string::npos,
                        "MTP loader rejects relative library paths");
    failures += require(!mtp.active(), "MTP loader remains inactive after path rejection");

    common_bridgespec_dflash_sidecar dflash;
    error.clear();
    failures += require(!dflash.load("relative-sidecar.so", "/absolute/artifacts", 25600, 8, 1, error) &&
                        error.find("absolute path") != std::string::npos,
                        "DFlash loader rejects relative library paths");
    failures += require(!dflash.active(), "DFlash loader remains inactive after path rejection");

    if (failures == 0) std::puts("artifact_manifest_test: PASS");
    return failures == 0 ? 0 : 1;
}
