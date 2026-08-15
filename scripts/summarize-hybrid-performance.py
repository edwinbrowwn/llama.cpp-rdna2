#!/usr/bin/env python3
"""Summarize one or more benchmark-hybrid-server JSON artifacts."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys


def mean(values):
    return statistics.fmean(values) if values else None


def stdev(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0 if values else None


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - index) + ordered[high] * (index - low)


def format_number(value, digits=3):
    return "" if value is None else f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for input_path in args.inputs:
        if input_path.is_dir():
            paths.extend(sorted(input_path.rglob("*.json")))
        else:
            paths.append(input_path)
    artifacts = []
    for path in paths:
        try:
            artifact = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if artifact.get("schema_version") != 1 or "groups" not in artifact:
            continue
        artifact["_path"] = str(path)
        artifacts.append(artifact)
    if not artifacts:
        parser.error("no benchmark artifacts found")

    rows = []
    for artifact in artifacts:
        gate = artifact.get("meta_fallback_gate", {})
        if gate.get("required") and not gate.get("passed"):
            raise RuntimeError(f"failed fallback gate in {artifact['_path']}: {gate}")
        for concurrency in artifact["concurrencies"]:
            groups = [group for group in artifact["groups"] if group["concurrency"] == concurrency]
            requests = [request for group in groups for request in group["requests"]]
            gpu_cards = sorted({card for group in groups for card in group.get("gpu", {})})
            gpu_busy = {
                card: mean(
                    [
                        group["gpu"][card]["mean_busy_percent"]
                        for group in groups
                        if card in group.get("gpu", {}) and group["gpu"][card]["mean_busy_percent"] is not None
                    ]
                )
                for card in gpu_cards
            }
            gpu_peak_vram = {
                card: max(
                    [
                        group["gpu"][card]["peak_vram_used_bytes"]
                        for group in groups
                        if card in group.get("gpu", {}) and group["gpu"][card]["peak_vram_used_bytes"] is not None
                    ],
                    default=None,
                )
                for card in gpu_cards
            }
            aggregate_gen = [
                group.get("aggregate_generation_tokens_per_second_window")
                or sum(int(request["timings"].get("predicted_n", 0)) for request in group["requests"])
                / (max(request["end_monotonic"] for request in group["requests"])
                   - min(request["first_token_monotonic"] for request in group["requests"]))
                for group in groups
            ]
            aggregate_e2e = [
                group.get("aggregate_output_tokens_per_second_end_to_end", group.get("aggregate_predicted_tokens_per_second"))
                for group in groups
            ]
            prompt_tps = [request["timings"].get("prompt_per_second") for request in requests]
            predicted_tps = [request["timings"].get("predicted_per_second") for request in requests]
            prompt_tps = [value for value in prompt_tps if value is not None]
            predicted_tps = [value for value in predicted_tps if value is not None]
            ttft = [request["ttft_seconds"] for request in requests]
            latency = [request["latency_seconds"] for request in requests]
            active_meta = [row for row in artifact.get("meta_copy_telemetry", []) if row["meta_attempts"]]
            specs = artifact.get("speculative_telemetry", [])
            # These are cumulative snapshots emitted as requests finish. The
            # row with the most generated tokens is the complete run snapshot.
            final_spec = max(specs, key=lambda row: row["generated_tokens"], default=None)
            generated_spec = final_spec["generated_tokens"] if final_spec else 0
            accepted_spec = final_spec["accepted_tokens"] if final_spec else 0
            rows.append(
                {
                    "label": artifact["label"],
                    "artifact": artifact["_path"],
                    "prompt_tokens": artifact["prompt_tokens"],
                    "n_predict": artifact["n_predict"],
                    "concurrency": concurrency,
                    "repeats": len(groups),
                    "aggregate_generation_tps_mean": mean(aggregate_gen),
                    "aggregate_generation_tps_stdev": stdev(aggregate_gen),
                    "aggregate_e2e_output_tps_mean": mean(aggregate_e2e),
                    "request_prompt_tps_mean": mean(prompt_tps),
                    "request_generation_tps_mean": mean(predicted_tps),
                    "ttft_seconds_mean": mean(ttft),
                    "ttft_seconds_p95": percentile(ttft, 0.95),
                    "latency_seconds_mean": mean(latency),
                    "latency_seconds_p95": percentile(latency, 0.95),
                    "gpu_mean_busy_percent": gpu_busy,
                    "gpu_peak_vram_bytes": gpu_peak_vram,
                    "meta_attempts_total_run": sum(row["meta_attempts"] for row in active_meta),
                    "meta_fallback_total_run": sum(row["meta_fallback"] for row in active_meta),
                    "meta_logical_bytes_total_run": sum(row["logical_bytes"] for row in active_meta),
                    "spec_generated_tokens_total_run": generated_spec,
                    "spec_accepted_tokens_total_run": accepted_spec,
                    "spec_acceptance_ratio_total_run": accepted_spec / generated_spec if generated_spec else None,
                    "communicator_sizes": artifact.get("communicator_sizes", []),
                    "pipeline_parallel_enabled": artifact.get("pipeline_parallel_enabled"),
                }
            )

    rows.sort(key=lambda row: (row["prompt_tokens"], row["n_predict"], row["concurrency"], row["label"]))
    (args.output_dir / "summary.json").write_text(json.dumps({"schema_version": 1, "rows": rows}, indent=2) + "\n")

    csv_fields = [
        "label", "prompt_tokens", "n_predict", "concurrency", "repeats",
        "aggregate_generation_tps_mean", "aggregate_generation_tps_stdev",
        "aggregate_e2e_output_tps_mean", "request_prompt_tps_mean",
        "request_generation_tps_mean", "ttft_seconds_mean", "ttft_seconds_p95",
        "latency_seconds_mean", "latency_seconds_p95", "meta_attempts_total_run",
        "meta_fallback_total_run", "meta_logical_bytes_total_run",
        "spec_generated_tokens_total_run", "spec_accepted_tokens_total_run",
        "spec_acceptance_ratio_total_run", "pipeline_parallel_enabled", "artifact",
    ]
    with (args.output_dir / "summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Hybrid server benchmark summary",
        "",
        "| configuration | PP | TG | C | aggregate generation tok/s | end-to-end output tok/s | request PP tok/s | request TG tok/s | TTFT mean/p95 s | latency mean/p95 s | MTP acceptance | Meta fallback |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        acceptance = "-" if row["spec_acceptance_ratio_total_run"] is None else format_number(row["spec_acceptance_ratio_total_run"], 4)
        lines.append(
            f"| {row['label']} | {row['prompt_tokens']} | {row['n_predict']} | {row['concurrency']} | "
            f"{format_number(row['aggregate_generation_tps_mean'])} ± {format_number(row['aggregate_generation_tps_stdev'])} | "
            f"{format_number(row['aggregate_e2e_output_tps_mean'])} | {format_number(row['request_prompt_tps_mean'])} | "
            f"{format_number(row['request_generation_tps_mean'])} | {format_number(row['ttft_seconds_mean'])} / {format_number(row['ttft_seconds_p95'])} | "
            f"{format_number(row['latency_seconds_mean'])} / {format_number(row['latency_seconds_p95'])} | {acceptance} | "
            f"{row['meta_fallback_total_run']} |"
        )
    lines.extend(
        [
            "",
            "Aggregate generation tok/s uses the interval from the first streamed token in a group to the last request completion. ",
            "End-to-end output tok/s includes prompt processing. Per-request PP/TG rates are the server-reported isolated timing fields. ",
            "Meta and MTP totals cover the complete server run (warm-up plus all measured groups), so they are repeated for each concurrency row.",
        ]
    )
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"HYBRID_PERFORMANCE_SUMMARY_OK artifacts={len(artifacts)} rows={len(rows)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"summary failed: {error}", file=sys.stderr)
        raise
