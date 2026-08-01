#!/usr/bin/env python3
"""Profile MoE expert routing with llama-imatrix.

The imatrix collector records the expert IDs seen by MUL_MAT_ID and writes
per-expert `.counts` tensors.  llama.cpp cannot currently place individual
experts from a fused 3-D expert tensor on different backends, so this tool
reports per-expert hotness and optionally emits a coarse whole-layer CPU
override as the safe CLI-level fallback.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - environment diagnostic
    raise SystemExit("numpy is required; use the Python environment shipped with gguf-py") from exc

NAME_RE = re.compile(r"^(?:blk|dspark)\.(\d+)\.(?:ffn_gate_exps|ffn_up_exps|ffn_down_exps)\.weight\.counts$")


def load_counts(path: Path) -> dict[int, list[int]]:
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "gguf-py"))
    try:
        from gguf import GGUFReader
    except ImportError as exc:
        raise SystemExit(f"cannot import gguf from {repo / 'gguf-py'}: {exc}") from exc

    reader = GGUFReader(str(path), "r")
    layers: dict[int, list[int]] = {}
    for tensor in reader.tensors:
        match = NAME_RE.match(tensor.name)
        if not match:
            continue
        values = np.asarray(tensor.data).reshape(-1)
        counts = [max(0, int(round(float(value)))) for value in values]
        layer = int(match.group(1))
        # gate/up/down counts should agree. Keep the largest observation if a
        # backend emitted duplicate statistics for the same fused expert tensor.
        if layer not in layers or sum(counts) > sum(layers[layer]):
            layers[layer] = counts
    if not layers:
        raise SystemExit(
            f"no fused expert .counts tensors found in {path}; "
            "ensure llama-imatrix ran on a MoE model with MUL_MAT_ID"
        )
    return dict(sorted(layers.items()))


def summarize(layers: dict[int, list[int]], coverage: float) -> dict[str, Any]:
    result: dict[str, Any] = {"coverage_target": coverage, "layers": {}}
    for layer, counts in layers.items():
        total = sum(counts)
        order = sorted(range(len(counts)), key=lambda i: (-counts[i], i))
        selected: list[int] = []
        running = 0
        target = total * coverage
        for expert in order:
            selected.append(expert)
            running += counts[expert]
            if running >= target:
                break
        result["layers"][str(layer)] = {
            "total_routes": total,
            "expert_count": len(counts),
            "nonzero_experts": sum(value > 0 for value in counts),
            "top_experts": order[: min(16, len(order))],
            "hot_experts_for_coverage": selected,
            "hot_routes": running,
            "counts": counts,
        }
    result["total_routes"] = sum(item["total_routes"] for item in result["layers"].values())
    return result


def make_layer_override(layers: dict[int, dict[str, Any]], cold_layer_count: int) -> str | None:
    if cold_layer_count <= 0:
        return None
    cold = sorted(layers, key=lambda layer: (layers[layer]["total_routes"], layer))[:cold_layer_count]
    if not cold:
        return None
    ids = "|".join(str(layer) for layer in cold)
    # This moves the entire routed expert tensor for selected layers. It does
    # not split individual fused experts; that is intentionally called out.
    return rf"blk\.({ids})\.ffn_(?:gate|up|down)_exps\.weight=CPU"


def write_launcher(path: Path, args: argparse.Namespace, override: str | None) -> None:
    model = shlex.quote(str(args.model))
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", f"MODEL={model}"]
    lines.append("# Expert counts are advisory; this is a whole-layer CPU fallback.")
    lines.append("# Individual fused experts cannot currently be assigned per-expert by -ot.")
    command = ["llama-server", "-m", '"$MODEL"', "-ngl", "all"]
    if override:
        command += ["--override-tensor", shlex.quote(override)]
    command += ["--flash-attn", "on"]
    lines.append(" \\\n  ".join(command))
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o755)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True, help="newline-separated representative prompts")
    parser.add_argument("--imatrix", type=Path, help="output/read existing imatrix GGUF")
    parser.add_argument("--imatrix-bin", type=Path, help="llama-imatrix executable")
    parser.add_argument("--ngl", default="all")
    parser.add_argument("--coverage", type=float, default=0.80)
    parser.add_argument("--cold-layers", type=int, default=0, help="coarse whole-layer CPU fallback")
    parser.add_argument("--report", type=Path, default=Path("expert-profile.json"))
    parser.add_argument("--launcher", type=Path, default=Path("run-profiled-moe.sh"))
    parser.add_argument("--keep-imatrix", action="store_true")
    args = parser.parse_args()

    if not args.model.is_file():
        parser.error(f"model not found: {args.model}")
    if not args.prompts.is_file():
        parser.error(f"prompt file not found: {args.prompts}")
    if not 0.0 < args.coverage <= 1.0:
        parser.error("--coverage must be in (0, 1]")

    imatrix = args.imatrix or args.report.with_suffix(".imatrix.gguf")
    if not imatrix.exists():
        binary = args.imatrix_bin or Path(__file__).resolve().parents[1] / "build/bin/llama-imatrix"
        if not binary.exists():
            binary = Path("llama-imatrix")
        command = [str(binary), "-m", str(args.model), "-f", str(args.prompts), "-o", str(imatrix),
                   "--output-format", "gguf", "--no-ppl", "-ngl", str(args.ngl)]
        print("profiling:", shlex.join(command), file=sys.stderr)
        subprocess.run(command, check=True)

    layers = load_counts(imatrix)
    report = summarize(layers, args.coverage)
    override = make_layer_override(report["layers"], args.cold_layers)
    report["source"] = {"model": str(args.model), "prompts": str(args.prompts), "imatrix": str(imatrix)}
    report["placement"] = {
        "individual_expert_cli_placement": False,
        "reason": "fused 3-D expert tensors are not independently addressable by --override-tensor",
        "coarse_cpu_override": override,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    write_launcher(args.launcher, args, override)

    print(f"wrote {args.report}")
    print(f"wrote {args.launcher}")
    for layer, item in report["layers"].items():
        print(f"layer {layer}: routes={item['total_routes']} nonzero={item['nonzero_experts']} hot={item['hot_experts_for_coverage']}")
    if override:
        print("coarse CPU override:", override)
    print("NOTE: per-expert GPU/CPU placement requires a runtime/GGUF layout change; this tool does not pretend -ot can do it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
