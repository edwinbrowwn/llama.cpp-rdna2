#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import statistics
import time
import urllib.request
from pathlib import Path


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    x = (len(ordered) - 1) * p
    lo = int(math.floor(x))
    hi = int(math.ceil(x))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - x) + ordered[hi] * (x - lo)


def rolling_min_rate(times_ms, window_ms):
    if len(times_ms) < 2:
        return None
    worst = None
    left = 0
    for right, t in enumerate(times_ms):
        while left < right and t - times_ms[left] > window_ms:
            left += 1
        elapsed = t - times_ms[left]
        if elapsed <= 0:
            continue
        rate = (right - left) * 1000.0 / elapsed
        worst = rate if worst is None else min(worst, rate)
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-predict", type=int, required=True)
    ap.add_argument("--seed", type=int, default=27742)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--top-p", type=float)
    ap.add_argument("--cache-prompt", action="store_true")
    ap.add_argument("--prompt-file")
    ap.add_argument("--prompt", default=(
        "Write an endless numbered list of concise facts about mathematics. "
        "Use one sentence per item and continue until interrupted.\n1."
    ))
    ap.add_argument("--spec-n-max", type=int)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prompt = Path(args.prompt_file).read_text() if args.prompt_file else args.prompt
    body = {
        "prompt": prompt,
        "n_predict": args.n_predict,
        "temperature": args.temperature,
        "top_p": args.top_p if args.top_p is not None else (0.95 if args.temperature > 0 else 1.0),
        "top_k": args.top_k if args.top_k is not None else (20 if args.temperature > 0 else 1),
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repeat_penalty": 1.0,
        "seed": args.seed,
        "cache_prompt": args.cache_prompt,
        "ignore_eos": True,
        "stream": True,
        "return_tokens": True,
    }
    if args.spec_n_max is not None:
        body["speculative.n_max"] = args.spec_n_max
    out.with_suffix(".request.json").write_text(json.dumps(body, indent=2) + "\n")

    req = urllib.request.Request(
        args.url.rstrip("/") + "/completion",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )

    start = time.perf_counter_ns()
    events = []
    token_arrivals = []
    token_ids = []
    output_parts = []
    final_payload = None
    with urllib.request.urlopen(req, timeout=args.timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")
        for raw in response:
            line = raw.strip()
            if not line or not line.startswith(b"data: "):
                continue
            now = time.perf_counter_ns()
            payload = json.loads(line[6:])
            rel_ms = (now - start) / 1e6
            tokens = payload.get("tokens") or []
            if isinstance(tokens, int):
                tokens = [tokens]
            events.append({
                "event": len(events),
                "relative_ms": rel_ms,
                "tokens": tokens,
                "content": payload.get("content", ""),
                "stop": bool(payload.get("stop", False)),
            })
            if tokens:
                # The server normally emits one token per SSE event. Preserve
                # zero gaps if an HTTP layer coalesces several accepted tokens.
                for token in tokens:
                    token_arrivals.append(rel_ms)
                    token_ids.append(int(token))
                output_parts.append(payload.get("content", ""))
            if payload.get("stop") or payload.get("timings"):
                final_payload = payload

    with out.with_suffix(".events.jsonl").open("w") as f:
        for event in events:
            f.write(json.dumps(event, separators=(",", ":")) + "\n")

    gaps = [token_arrivals[i] - token_arrivals[i - 1] for i in range(1, len(token_arrivals))]
    top_gaps = sorted(
        ({"token_index": i, "gap_ms": gaps[i - 1], "token": token_ids[i]}
         for i in range(1, len(token_arrivals))),
        key=lambda x: x["gap_ms"], reverse=True,
    )[:64]
    timings = (final_payload or {}).get("timings") or {}
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    arrival_span = token_arrivals[-1] - token_arrivals[0] if len(token_arrivals) > 1 else 0.0
    output = "".join(output_parts)
    gap_summary = {
        "count": len(gaps),
        "min": min(gaps) if gaps else None,
        "median": statistics.median(gaps) if gaps else None,
        "mean": statistics.mean(gaps) if gaps else None,
        "p90": percentile(gaps, 0.90),
        "p95": percentile(gaps, 0.95),
        "p99": percentile(gaps, 0.99),
        "p999": percentile(gaps, 0.999),
        "max": max(gaps) if gaps else None,
        "le_1ms": sum(x <= 1.0 for x in gaps),
        "gt_25ms": sum(x > 25.0 for x in gaps),
        "gt_50ms": sum(x > 50.0 for x in gaps),
        "gt_100ms": sum(x > 100.0 for x in gaps),
        "gt_200ms": sum(x > 200.0 for x in gaps),
    }
    summary = {
        "status": "pass" if len(token_ids) == args.n_predict else "unexpected_token_count",
        "requested_tokens": args.n_predict,
        "received_tokens": len(token_ids),
        "token_events": sum(bool(e["tokens"]) for e in events),
        "all_events": len(events),
        "elapsed_ms": elapsed_ms,
        "ttft_ms": token_arrivals[0] if token_arrivals else None,
        "arrival_tokens_per_second": ((len(token_ids) - 1) * 1000.0 / arrival_span
                                      if arrival_span > 0 else None),
        "rolling_min_tokens_per_second": {
            "1000ms": rolling_min_rate(token_arrivals, 1000.0),
            "3000ms": rolling_min_rate(token_arrivals, 3000.0),
        },
        "server_timings": timings,
        "gap_ms": gap_summary,
        "top_gaps": top_gaps,
        "token_sha256": hashlib.sha256(",".join(map(str, token_ids)).encode()).hexdigest(),
        "text_sha256": hashlib.sha256(output.encode()).hexdigest(),
    }
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    out.with_suffix(".tokens.json").write_text(json.dumps(token_ids) + "\n")
    out.with_suffix(".txt").write_text(output)
    print(json.dumps({
        "status": summary["status"],
        "received_tokens": summary["received_tokens"],
        "predicted_tps": timings.get("predicted_per_second"),
        "draft_n": timings.get("draft_n", 0),
        "draft_n_accepted": timings.get("draft_n_accepted", 0),
        "arrival_tps": summary["arrival_tokens_per_second"],
        "rolling_min_tps": summary["rolling_min_tokens_per_second"],
        "gap_ms": gap_summary,
        "token_sha256": summary["token_sha256"],
    }, sort_keys=True))
    if summary["status"] != "pass":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
