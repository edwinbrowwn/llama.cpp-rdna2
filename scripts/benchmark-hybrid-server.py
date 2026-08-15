#!/usr/bin/env python3
"""Run bounded, repeatable llama-server concurrency benchmarks.

The script launches one server configuration, creates exact-length token-array
prompts through /tokenize, streams /completion requests to measure TTFT and
latency, samples per-GPU sysfs activity, and fails closed on Meta PP fallback
when requested.
"""

import argparse
import concurrent.futures
import glob
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


def http_json(url, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data)
    if payload is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def stream_completion(url, payload, barrier, timeout):
    barrier.wait()
    start = time.monotonic()
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    content = []
    token_ids = []
    first_token = None
    final = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or line == "data: [DONE]":
                continue
            if not line.startswith("data: "):
                continue
            chunk = json.loads(line[6:])
            final = chunk
            piece = chunk.get("content") or ""
            tokens = chunk.get("tokens") or []
            if first_token is None and (piece or tokens):
                first_token = time.monotonic()
            content.append(piece)
            token_ids.extend(tokens)
    end = time.monotonic()
    if final is None:
        raise RuntimeError("stream returned no JSON chunks")
    if first_token is None:
        first_token = end
    return {
        "start_monotonic": start,
        "first_token_monotonic": first_token,
        "end_monotonic": end,
        "ttft_seconds": first_token - start,
        "latency_seconds": end - start,
        "content": "".join(content),
        "tokens": token_ids,
        "stop_type": final.get("stop_type"),
        "timings": final.get("timings") or {},
        "truncated": final.get("truncated"),
    }


def read_int(path):
    try:
        return int(Path(path).read_text().strip())
    except (FileNotFoundError, PermissionError, ValueError, OSError):
        return None


def discover_gpus():
    devices = []
    for card_path in sorted(glob.glob("/sys/class/drm/card[0-9]*")):
        card = Path(card_path)
        vendor_path = card / "device/vendor"
        if not vendor_path.is_file() or vendor_path.read_text().strip().lower() != "0x1002":
            continue
        device = card / "device"
        hwmons = sorted((device / "hwmon").glob("hwmon*"))
        devices.append(
            {
                "card": card.name,
                "bdf": device.resolve().name,
                "busy": device / "gpu_busy_percent",
                "vram_used": device / "mem_info_vram_used",
                "vram_total": device / "mem_info_vram_total",
                "temps": [
                    temp
                    for hwmon in hwmons
                    for temp in sorted(hwmon.glob("temp*_input"))
                ],
            }
        )
    return devices


def sample_gpus(devices, stop_event, samples, interval):
    while not stop_event.is_set():
        now = time.monotonic()
        row = {"monotonic": now, "devices": {}}
        for device in devices:
            temps = [value for path in device["temps"] if (value := read_int(path)) is not None]
            row["devices"][device["card"]] = {
                "bdf": device["bdf"],
                "busy_percent": read_int(device["busy"]),
                "vram_used_bytes": read_int(device["vram_used"]),
                "vram_total_bytes": read_int(device["vram_total"]),
                "max_temp_millic": max(temps) if temps else None,
            }
        samples.append(row)
        stop_event.wait(interval)


def summarize_gpu_samples(samples, start, end):
    selected = [row for row in samples if start <= row["monotonic"] <= end]
    cards = sorted({card for row in selected for card in row["devices"]})
    output = {}
    for card in cards:
        values = [row["devices"][card] for row in selected if card in row["devices"]]
        busy = [value["busy_percent"] for value in values if value["busy_percent"] is not None]
        vram = [value["vram_used_bytes"] for value in values if value["vram_used_bytes"] is not None]
        temps = [value["max_temp_millic"] for value in values if value["max_temp_millic"] is not None]
        output[card] = {
            "bdf": values[0]["bdf"] if values else None,
            "samples": len(values),
            "mean_busy_percent": sum(busy) / len(busy) if busy else None,
            "peak_busy_percent": max(busy) if busy else None,
            "peak_vram_used_bytes": max(vram) if vram else None,
            "peak_temp_millic": max(temps) if temps else None,
        }
    return output


def make_prompt_tokens(base_url, target, variant):
    paragraph = (
        f"Request variant {variant}. Pipeline parallel inference should preserve numerical correctness, "
        "sequence state, and deterministic token order while independent requests execute concurrently. "
        "This natural-language paragraph is repeated to create a controlled prompt-processing workload. "
    )
    text = paragraph
    while True:
        response = http_json(
            base_url + "/tokenize",
            {"content": text, "add_special": True, "parse_special": True},
            timeout=60,
        )
        tokens = response["tokens"]
        if len(tokens) >= target:
            return tokens[:target]
        text += paragraph


def wait_for_server(base_url, process, timeout):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup with status {process.returncode}")
        try:
            health = http_json(base_url + "/health", timeout=2)
            if health.get("status") in ("ok", "no slot available"):
                return health
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
        time.sleep(0.25)
    raise RuntimeError(f"server health timeout: {last_error}")


def parse_meta_telemetry(log_text):
    pattern = re.compile(
        r"META_COPY summary backend=(.*?) attempts=(\d+) meta_attempts=(\d+) success=(\d+) "
        r"fallback=(\d+) meta_fallback=(\d+) unsupported_state=(\d+) "
        r"logical_bytes=(\d+) physical_bytes=(\d+)"
    )
    rows = []
    for match in pattern.finditer(log_text):
        rows.append(
            {
                "backend": match.group(1),
                "attempts": int(match.group(2)),
                "meta_attempts": int(match.group(3)),
                "success": int(match.group(4)),
                "fallback": int(match.group(5)),
                "meta_fallback": int(match.group(6)),
                "unsupported_state": int(match.group(7)),
                "logical_bytes": int(match.group(8)),
                "physical_bytes": int(match.group(9)),
            }
        )
    return rows


def parse_speculative_telemetry(log_text):
    pattern = re.compile(
        r"statistics\s+(\S+): #calls\(b,g,a\) =\s+(\d+)\s+(\d+)\s+(\d+), "
        r"#gen drafts =\s+(\d+), #acc drafts =\s+(\d+), #gen tokens =\s+(\d+), "
        r"#acc tokens =\s+(\d+), #mean acc len =\s+([0-9.]+), #acc rate/pos = \(([^)]*)\), "
        r"dur\(b,g,a\) = ([0-9.]+), ([0-9.]+), ([0-9.]+) ms"
    )
    rows = []
    for match in pattern.finditer(log_text):
        rows.append(
            {
                "type": match.group(1),
                "calls_begin": int(match.group(2)),
                "calls_generate": int(match.group(3)),
                "calls_accept": int(match.group(4)),
                "generated_drafts": int(match.group(5)),
                "accepted_drafts": int(match.group(6)),
                "generated_tokens": int(match.group(7)),
                "accepted_tokens": int(match.group(8)),
                "mean_accepted_length": float(match.group(9)),
                "accept_rate_by_position": [float(value.strip()) for value in match.group(10).split(",")],
                "duration_begin_ms": float(match.group(11)),
                "duration_generate_ms": float(match.group(12)),
                "duration_accept_ms": float(match.group(13)),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--concurrency", default="1,2")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--n-predict", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--startup-timeout", type=float, default=240.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.25)
    parser.add_argument("--seed-base", type=int, default=20260815)
    parser.add_argument("--require-meta-zero-fallback", action="store_true")
    parser.add_argument("server_command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = list(args.server_command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("server command is required after --")
    concurrencies = [int(value) for value in args.concurrency.split(",")]
    if any(value <= 0 for value in concurrencies):
        parser.error("concurrency values must be positive")
    if args.prompt_tokens <= 0 or args.n_predict <= 0 or args.repeats <= 0:
        parser.error("prompt, generation, and repeat counts must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / f"{args.label}.server.log"
    result_path = args.output_dir / f"{args.label}.json"
    base_url = f"http://127.0.0.1:{args.port}"
    command.extend(["--host", "127.0.0.1", "--port", str(args.port)])

    devices = discover_gpus()
    gpu_samples = []
    stop_samples = threading.Event()
    sampler = threading.Thread(
        target=sample_gpus,
        args=(devices, stop_samples, gpu_samples, args.gpu_sample_interval),
        daemon=True,
    )

    with log_path.open("w") as log_file:
        process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
        try:
            health = wait_for_server(base_url, process, args.startup_timeout)
            try:
                server_version = http_json(base_url + "/version", timeout=10)
            except urllib.error.HTTPError as error:
                if error.code != 404:
                    raise
                server_version = None
            prompts = {
                concurrency: [
                    make_prompt_tokens(base_url, args.prompt_tokens, index)
                    for index in range(concurrency)
                ]
                for concurrency in concurrencies
            }
            sampler.start()

            # Bounded warm-up outside measured groups.
            warm_tokens = prompts[concurrencies[0]][0][: min(32, args.prompt_tokens)]
            warm_barrier = threading.Barrier(1)
            stream_completion(
                base_url + "/completion",
                {
                    "prompt": warm_tokens,
                    "n_predict": min(8, args.n_predict),
                    "temperature": 0,
                    "seed": args.seed_base - 1,
                    "ignore_eos": True,
                    "cache_prompt": False,
                    "stream": True,
                    "return_tokens": True,
                },
                warm_barrier,
                args.request_timeout,
            )

            groups = []
            for concurrency in concurrencies:
                for repeat in range(args.repeats):
                    barrier = threading.Barrier(concurrency)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                        futures = [
                            executor.submit(
                                stream_completion,
                                base_url + "/completion",
                                {
                                    "prompt": prompts[concurrency][index],
                                    "n_predict": args.n_predict,
                                    "temperature": 0,
                                    "seed": args.seed_base + index,
                                    "ignore_eos": True,
                                    "cache_prompt": False,
                                    "stream": True,
                                    "return_tokens": True,
                                },
                                barrier,
                                args.request_timeout,
                            )
                            for index in range(concurrency)
                        ]
                        requests = [future.result() for future in futures]
                    group_start = min(request["start_monotonic"] for request in requests)
                    group_end = max(request["end_monotonic"] for request in requests)
                    wall = group_end - group_start
                    generation_start = min(request["first_token_monotonic"] for request in requests)
                    prompt_end = max(request["first_token_monotonic"] for request in requests)
                    predicted = sum(int(request["timings"].get("predicted_n", 0)) for request in requests)
                    prompt_n = sum(int(request["timings"].get("prompt_n", 0)) for request in requests)
                    groups.append(
                        {
                            "concurrency": concurrency,
                            "repeat": repeat,
                            "wall_seconds": wall,
                            "prompt_window_seconds": prompt_end - group_start,
                            "generation_window_seconds": group_end - generation_start,
                            "aggregate_output_tokens_per_second_end_to_end": predicted / wall,
                            "aggregate_generation_tokens_per_second_window": predicted / (group_end - generation_start),
                            "aggregate_prompt_tokens_per_second_window": prompt_n / (prompt_end - group_start),
                            "requests": requests,
                            "gpu_window": [group_start, group_end],
                        }
                    )
        finally:
            stop_samples.set()
            if sampler.is_alive():
                sampler.join(timeout=5)
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=15)

    log_text = log_path.read_text(errors="replace")
    telemetry = parse_meta_telemetry(log_text)
    for group in groups:
        group["gpu"] = summarize_gpu_samples(gpu_samples, *group.pop("gpu_window"))
    result = {
        "schema_version": 1,
        "label": args.label,
        "command": command,
        "environment": {
            key: os.environ.get(key)
            for key in [
                "HSA_OVERRIDE_GFX_VERSION",
                "HSA_NO_SCRATCH_RECLAIM",
                "GGML_CUDA_DISABLE_GRAPHS",
                "GGML_CUDA_ALLREDUCE",
                "GGML_META_COPY_DEBUG",
                "GGML_SCHED_TRACE",
                "GGML_TP_SHARDED_OUTPUT",
                "LD_LIBRARY_PATH",
            ]
        },
        "prompt_tokens": args.prompt_tokens,
        "n_predict": args.n_predict,
        "seed_base": args.seed_base,
        "health": health,
        "server_version": server_version,
        "repeats": args.repeats,
        "concurrencies": concurrencies,
        "groups": groups,
        "meta_copy_telemetry": telemetry,
        "speculative_telemetry": parse_speculative_telemetry(log_text),
        "communicator_sizes": [int(value) for value in re.findall(r"RCCL/NCCL AllReduce across (\d+) devices", log_text)],
        "pipeline_stages": [
            {
                "stage": int(match.group(1)),
                "backend": match.group(2),
                "first_layer": int(match.group(3)),
                "last_layer": int(match.group(4)),
                "owns_auxiliary_and_output": "+ auxiliary/NextN + output" in match.group(5),
            }
            for match in re.finditer(
                r"parallel: stage (\d+) = (.*?), transformer layers (\d+)\.\.(\d+)(.*)",
                log_text,
            )
        ],
        "pipeline_parallel_enabled": "pipeline parallelism enabled" in log_text,
        "gpu_samples": gpu_samples,
    }

    gate_error = None
    if args.require_meta_zero_fallback:
        active = [row for row in telemetry if row["meta_attempts"] > 0]
        if not active:
            gate_error = "no active Meta-to-Meta telemetry summary found"
        for row in active:
            if row["success"] != row["meta_attempts"] or row["meta_fallback"] != 0 or row["unsupported_state"] != 0:
                gate_error = f"Meta PP fallback gate failed: {row}"
                break
            if row["physical_bytes"] != 2 * row["logical_bytes"]:
                gate_error = f"unexpected two-rank byte accounting: {row}"
                break
    result["meta_fallback_gate"] = {
        "required": args.require_meta_zero_fallback,
        "passed": gate_error is None,
        "error": gate_error,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    if gate_error is not None:
        raise RuntimeError(gate_error)
    compact = []
    for concurrency in concurrencies:
        selected = [group for group in groups if group["concurrency"] == concurrency]
        compact.append(
            {
                "concurrency": concurrency,
                "aggregate_generation_tps_mean": sum(group["aggregate_generation_tokens_per_second_window"] for group in selected) / len(selected),
                "aggregate_end_to_end_output_tps_mean": sum(group["aggregate_output_tokens_per_second_end_to_end"] for group in selected) / len(selected),
                "ttft_seconds_mean": sum(request["ttft_seconds"] for group in selected for request in group["requests"]) / sum(len(group["requests"]) for group in selected),
                "latency_seconds_mean": sum(request["latency_seconds"] for group in selected for request in group["requests"]) / sum(len(group["requests"]) for group in selected),
            }
        )
    print("HYBRID_SERVER_BENCH_OK", json.dumps({"label": args.label, "results": compact}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"benchmark failed: {error}", file=sys.stderr)
        raise
