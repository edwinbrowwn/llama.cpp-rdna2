#!/usr/bin/env python3
import argparse
import collections
import json
import math
import re
from pathlib import Path


def pct(values, p):
    if not values:
        return None
    values = sorted(values)
    x = (len(values) - 1) * p
    lo = math.floor(x); hi = math.ceil(x)
    return values[lo] if lo == hi else values[lo] * (hi - x) + values[hi] * (x - lo)


def dist(values):
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": pct(values, .50),
        "p95": pct(values, .95),
        "p99": pct(values, .99),
        "max": max(values) if values else None,
    }


def longest_streak(rows, predicate):
    best = cur = 0
    for row in rows:
        if predicate(row):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def row_summary(rows):
    first_reject = sum(r.get("first_reject") == 1 for r in rows)
    accepted_hist = collections.Counter(str(r.get("accepted", -1)) for r in rows)
    proposed_hist = collections.Counter(str(r.get("proposed", -1)) for r in rows)
    cycle_total_us = sum(r.get("cycle_us", 0) for r in rows)
    return {
        "cycles": len(rows),
        "accepted_histogram": dict(sorted(accepted_hist.items(), key=lambda x: int(x[0]))),
        "proposed_histogram": dict(sorted(proposed_hist.items(), key=lambda x: int(x[0]))),
        "first_reject_cycles": first_reject,
        "first_reject_rate": first_reject / len(rows) if rows else None,
        "longest_first_reject_streak": longest_streak(rows, lambda r: r.get("first_reject") == 1),
        "accepted_total": sum(r.get("accepted", 0) for r in rows),
        "proposed_total": sum(r.get("proposed", 0) for r in rows),
        "emitted_total": sum(r.get("emitted", 0) for r in rows),
        "effective_emitted_tps": (sum(r.get("emitted", 0) for r in rows) * 1e6 / cycle_total_us
                                  if cycle_total_us > 0 else None),
        "draft_us": dist([r["draft_us"] for r in rows if "draft_us" in r]),
        "verify_us": dist([r["verify_us"] for r in rows if "verify_us" in r]),
        "cycle_us": dist([r["cycle_us"] for r in rows if "cycle_us" in r]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup-cycles", type=int, default=2)
    a = ap.parse_args()
    rows = []
    for line_no, line in enumerate(Path(a.log).read_text(errors="replace").splitlines(), 1):
        marker = line.find("Q2CYCLE ")
        if marker < 0:
            continue
        fields = dict(re.findall(r"([a-z_]+)=([^\s]+)", line[marker + len("Q2CYCLE "):]))
        row = {"line": line_no, "raw": line}
        for key, value in fields.items():
            if key == "source":
                row[key] = value
            else:
                try: row[key] = int(value)
                except ValueError: row[key] = value
        rows.append(row)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix(".jsonl").open("w") as f:
        for row in rows: f.write(json.dumps(row, separators=(",", ":")) + "\n")
    by_source = collections.Counter(r.get("source", "unknown") for r in rows)
    steady = rows[max(0, a.warmup_cycles):]
    fixed4 = [r for r in steady if r.get("proposed") == 4]
    widths = {}
    for width in sorted({r.get("proposed") for r in steady}):
        widths[str(width)] = row_summary([r for r in steady if r.get("proposed") == width])
    summary = row_summary(rows)
    summary.update({
        "source_counts": dict(sorted(by_source.items())),
        "warmup_cycles_excluded": min(len(rows), max(0, a.warmup_cycles)),
        "steady": row_summary(steady),
        "steady_fixed4": row_summary(fixed4),
        "by_proposed_width": widths,
        "graph_reused_delta": dict(collections.Counter(
            str(r.get("graph_reused_delta", -1)) for r in rows)),
    })
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
