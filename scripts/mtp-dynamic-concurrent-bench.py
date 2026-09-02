#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import random
import statistics
import threading
import time
import urllib.request
from pathlib import Path


def make_prompt(base_url: str, slot: int, length: int, salt: int, prompt_kind: str) -> list[int]:
    # Build synthetic prompts from ordinary text tokens. Uniform random token IDs
    # can decode to malformed UTF-8 and exercise response-parser error handling
    # instead of model throughput.
    words = (
        "analysis system memory bandwidth latency throughput scheduler kernel model token "
        "context cache request response parallel compute device stream tensor matrix vector "
        "network storage process thread queue measurement benchmark reliable consistent "
        "runtime service workload capacity efficiency sequence prompt generation attention "
        "verification sampling result operation architecture performance resource concurrent"
    ).split()
    rng = random.Random(0x5EED0000 + salt * 1009 + slot * 7919)
    if prompt_kind == "repeat":
        phrase = "amber cobalt jade ivory onyx silver copper violet. "
        text = (phrase * max(16, length // 4 + 8)).strip()
    else:
        text = f"Benchmark user {slot} phase {salt}. " + " ".join(
            rng.choice(words) for _ in range(max(64, length * 2))
        )
    req = urllib.request.Request(
        base_url + "/tokenize",
        data=json.dumps({"content": text, "add_special": True, "parse_special": True}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        tokens = json.load(response)["tokens"]
    if len(tokens) < length:
        raise RuntimeError(f"slot {slot}: tokenizer produced only {len(tokens)} tokens, need {length}")
    return tokens[:length]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def gap_summary(values: list[float]) -> dict:
    return {
        "count": len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else 0.0,
        "gt_50ms": sum(v > 50.0 for v in values),
        "gt_100ms": sum(v > 100.0 for v in values),
        "gt_200ms": sum(v > 200.0 for v in values),
    }


def run_phase(base_url: str, parallel: int, phase: str, prompt_len: int, n_predict: int, salt: int, prompt_kind: str) -> dict:
    barrier = threading.Barrier(parallel + 1)
    prompts = [make_prompt(base_url, slot, prompt_len, salt, prompt_kind) for slot in range(parallel)]

    def run_one(slot: int) -> dict:
        payload = {
            "prompt": prompts[slot],
            "n_predict": n_predict,
            "ignore_eos": True,
            "cache_prompt": False,
            "stream": True,
            "reasoning_format": "none",
            "seed": 424242 + salt * 1000 + slot,
            "id_slot": slot,
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            base_url + "/completion",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        barrier.wait()
        submitted = time.perf_counter()
        first_token = None
        last_token = None
        timings = None
        stream_tokens = 0
        token_arrivals = []
        final = None
        with urllib.request.urlopen(req, timeout=1800) as response:
            for raw in response:
                if not raw.startswith(b"data: "):
                    continue
                obj = json.loads(raw[6:])
                now = time.perf_counter()
                final = obj
                if "timings" in obj:
                    timings = obj["timings"]
                tokens = obj.get("tokens") or []
                is_token = bool(tokens) or (not obj.get("stop", False) and bool(obj.get("content", "")))
                if is_token:
                    if first_token is None:
                        first_token = now
                    last_token = now
                    token_arrivals.extend([now] * (len(tokens) if tokens else 1))
                    stream_tokens += len(tokens) if tokens else 1
        ended = time.perf_counter()
        if timings is None and isinstance(final, dict):
            timings = final.get("timings")
        if timings is None:
            raise RuntimeError(f"slot {slot}: no timings in streaming response")
        if first_token is None:
            first_token = ended
            last_token = ended
        token_gaps_ms = [1000.0 * (b - a) for a, b in zip(token_arrivals, token_arrivals[1:])]
        return {
            "slot": slot,
            "submitted": submitted,
            "first_token": first_token,
            "last_token": last_token,
            "ended": ended,
            "stream_tokens": stream_tokens,
            "token_gap_ms": token_gaps_ms,
            "gap_ms": gap_summary(token_gaps_ms),
            "timings": timings,
        }

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = [pool.submit(run_one, i) for i in range(parallel)]
        barrier.wait()
        results = [f.result() for f in futures]
    finished = time.perf_counter()

    total_prompt = sum(int(r["timings"].get("prompt_n", 0)) for r in results)
    total_predicted = sum(int(r["timings"].get("predicted_n", 0)) for r in results)
    total_drafted = sum(int(r["timings"].get("draft_n", 0)) for r in results)
    total_accepted = sum(int(r["timings"].get("draft_n_accepted", 0)) for r in results)

    # Reconstruct phase windows from each request's server-reported durations and
    # its response completion timestamp. This accounts for slightly staggered slots.
    prompt_starts = []
    prompt_ends = []
    generation_starts = []
    generation_ends = []
    for r in results:
        prompt_s = float(r["timings"].get("prompt_ms", 0.0)) / 1000.0
        generation_s = float(r["timings"].get("predicted_ms", 0.0)) / 1000.0
        generation_end = r["ended"]
        generation_start = generation_end - generation_s
        prompt_end = generation_start
        prompt_start = prompt_end - prompt_s
        prompt_starts.append(prompt_start)
        prompt_ends.append(prompt_end)
        generation_starts.append(generation_start)
        generation_ends.append(generation_end)

    prompt_window = max(prompt_ends) - min(prompt_starts)
    generation_window = max(generation_ends) - min(generation_starts)
    client_wall = max(r["ended"] for r in results) - min(r["submitted"] for r in results)
    ttft_window = max(r["first_token"] for r in results) - min(r["submitted"] for r in results)

    all_gaps = [gap for result in results for gap in result["token_gap_ms"]]

    return {
        "phase": phase,
        "parallel": parallel,
        "prompt_len_requested": prompt_len,
        "n_predict_requested": n_predict,
        "total_prompt_tokens": total_prompt,
        "total_predicted_tokens": total_predicted,
        "total_drafted_tokens": total_drafted,
        "total_accepted_tokens": total_accepted,
        "acceptance": total_accepted / total_drafted if total_drafted else 0.0,
        "server_prompt_window_s": prompt_window,
        "server_generation_window_s": generation_window,
        "aggregate_prompt_tps": total_prompt / prompt_window if prompt_window > 0 else 0.0,
        "aggregate_generation_tps": total_predicted / generation_window if generation_window > 0 else 0.0,
        "client_wall_s": client_wall,
        "client_end_to_end_output_tps": total_predicted / client_wall if client_wall > 0 else 0.0,
        "client_ttft_window_s": ttft_window,
        "mean_ttft_ms": 1000.0 * statistics.mean(r["first_token"] - r["submitted"] for r in results),
        "gap_ms": gap_summary(all_gaps),
        "results": results,
        "orchestration_wall_s": finished - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--parallel", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-predict", type=int, default=128)
    parser.add_argument("--prompt-kind", choices=("ordinary", "repeat"), default="ordinary")
    args = parser.parse_args()

    phases = []
    phases.append(run_phase(args.url, args.parallel, "warmup", 512, 8, 1, args.prompt_kind))
    time.sleep(1.0)
    phases.append(run_phase(args.url, args.parallel, "pp", 512, 1, 2, args.prompt_kind))
    time.sleep(1.0)
    tg_prompt = 128 if args.prompt_kind == "repeat" else 32
    phases.append(run_phase(args.url, args.parallel, "tg", tg_prompt, args.n_predict, 3, args.prompt_kind))

    report = {
        "parallel": args.parallel,
        "method": {
            "pp": "sum prompt tokens / reconstructed concurrent prompt window",
            "tg": "sum generated tokens / reconstructed concurrent generation window",
            "requests": "one pinned request per slot; cache_prompt=false; fixed-length tokenized text",
            "prompt_kind": args.prompt_kind,
        },
        "phases": phases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    pp = next(p for p in phases if p["phase"] == "pp")
    tg = next(p for p in phases if p["phase"] == "tg")
    summary = {
        "parallel": args.parallel,
        "aggregate_pp_tps": pp["aggregate_prompt_tps"],
        "pp_tokens": pp["total_prompt_tokens"],
        "pp_window_s": pp["server_prompt_window_s"],
        "aggregate_tg_tps": tg["aggregate_generation_tps"],
        "tg_tokens": tg["total_predicted_tokens"],
        "tg_window_s": tg["server_generation_window_s"],
        "tg_acceptance": tg["acceptance"],
        "tg_drafted": tg["total_drafted_tokens"],
        "tg_accepted": tg["total_accepted_tokens"],
        "tg_client_e2e_tps": tg["client_end_to_end_output_tps"],
        "tg_wall_s": tg["client_wall_s"],
        "tg_gap_ms": tg["gap_ms"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
