#!/usr/bin/env python3
"""Correlate GGML_SCHED_TRACE stage enqueue ranges with rocprofv3 GPU kernels.

The scheduler emits CLOCK_MONOTONIC microsecond timestamps. rocprofv3 emits
nanoseconds in the same clock domain. HIP API correlation IDs associate host
calls made inside each stage enqueue range with completed GPU kernel records.
"""

import argparse
import bisect
import collections
import json
import re
from pathlib import Path

BEGIN_RE = re.compile(
    r"stage_enqueue_begin dispatch=(\d+) slot=(\d+) split=(\d+) "
    r"backend='([^']+)' nodes=(\d+) ts_us=(\d+)"
)
END_RE = re.compile(
    r"stage_enqueue_end dispatch=(\d+) slot=(\d+) split=(\d+) "
    r"backend='([^']+)' ts_us=(\d+)"
)


def parse_stage_intervals(log_path: Path):
    pending = collections.defaultdict(collections.deque)
    intervals = []
    order = 0
    for line in log_path.read_text(errors="replace").splitlines():
        match = BEGIN_RE.search(line)
        if match:
            dispatch, slot, split, backend, nodes, timestamp = match.groups()
            key = (int(dispatch), int(split), backend)
            pending[key].append(
                {
                    "dispatch": int(dispatch),
                    "slot": int(slot),
                    "split": int(split),
                    "backend": backend,
                    "nodes": int(nodes),
                    "host_start_ns": int(timestamp) * 1000,
                    "order": order,
                }
            )
            order += 1
            continue

        match = END_RE.search(line)
        if match:
            dispatch, slot, split, backend, timestamp = match.groups()
            key = (int(dispatch), int(split), backend)
            if not pending[key]:
                raise RuntimeError(f"unmatched stage end: {line}")
            interval = pending[key].popleft()
            if interval["slot"] != int(slot):
                raise RuntimeError(f"slot changed inside stage interval: {line}")
            interval["host_end_ns"] = int(timestamp) * 1000
            intervals.append(interval)

    unmatched = sum(len(queue) for queue in pending.values())
    if unmatched:
        raise RuntimeError(f"{unmatched} unmatched stage begin records")
    return sorted(intervals, key=lambda item: item["host_start_ns"])


def union_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def intersection_ns(left, right):
    i = 0
    j = 0
    total = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end > start:
            total += end - start
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def load_tool_record(trace_path: Path):
    document = json.loads(trace_path.read_text())
    tools = document.get("rocprofiler-sdk-tool")
    if not isinstance(tools, list) or len(tools) != 1:
        raise RuntimeError("expected one rocprofiler-sdk-tool record")
    return tools[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage0-backend")
    parser.add_argument("--stage1-backend")
    args = parser.parse_args()

    stages = [
        interval
        for interval in parse_stage_intervals(args.log)
        if interval["backend"].startswith("Meta(")
    ]
    if not stages:
        raise RuntimeError("no Meta stage enqueue intervals found")

    tool = load_tool_record(args.trace)
    hip_records = tool["buffer_records"]["hip_api"]
    kernel_records = tool["buffer_records"]["kernel_dispatch"]

    stage_starts = [stage["host_start_ns"] for stage in stages]
    correlation_to_stage = {}
    for record in hip_records:
        timestamp = record["start_timestamp"]
        index = bisect.bisect_right(stage_starts, timestamp) - 1
        if index >= 0 and timestamp <= stages[index]["host_end_ns"]:
            correlation_to_stage[record["correlation_id"]["internal"]] = index

    kernels_by_stage = collections.defaultdict(list)
    for kernel in kernel_records:
        index = correlation_to_stage.get(kernel["correlation_id"]["internal"])
        if index is not None:
            kernels_by_stage[index].append(kernel)

    stage_rows = []
    for index, stage in enumerate(stages):
        kernels = kernels_by_stage[index]
        if not kernels:
            continue
        device_intervals = [
            (kernel["start_timestamp"], kernel["end_timestamp"])
            for kernel in kernels
        ]
        busy = union_intervals(device_intervals)
        row = dict(stage)
        row.update(
            {
                "kernel_count": len(kernels),
                "agents": sorted(
                    {
                        kernel["dispatch_info"]["agent_id"]["handle"]
                        for kernel in kernels
                    }
                ),
                "device_start_ns": min(start for start, _ in device_intervals),
                "device_end_ns": max(end for _, end in device_intervals),
                "device_busy_ns": sum(end - start for start, end in busy),
                "busy_intervals": busy,
            }
        )
        stage_rows.append(row)

    backend_order = []
    for stage in stages:
        if stage["backend"] not in backend_order:
            backend_order.append(stage["backend"])
    stage0_backend = args.stage0_backend or (backend_order[0] if backend_order else None)
    stage1_backend = args.stage1_backend or (backend_order[1] if len(backend_order) > 1 else None)
    if stage0_backend is None or stage1_backend is None or stage0_backend == stage1_backend:
        raise RuntimeError("expected two distinct traced Meta stage backends")

    stage_zero = [row for row in stage_rows if row["backend"] == stage0_backend]
    stage_one = [row for row in stage_rows if row["backend"] == stage1_backend]
    if not stage_zero or not stage_one:
        raise RuntimeError(
            f"missing traced stage kernels for {stage0_backend!r} or {stage1_backend!r}"
        )

    overlap_pairs = []
    for later in stage_zero:
        previous = [
            earlier
            for earlier in stage_one
            if earlier["host_start_ns"] < later["host_start_ns"]
        ]
        if not previous:
            continue
        earlier = previous[-1]
        overlap = intersection_ns(earlier["busy_intervals"], later["busy_intervals"])
        overlap_pairs.append(
            {
                "stage1_a_order": earlier["order"],
                "stage1_a_dispatch": earlier["dispatch"],
                "stage1_a_slot": earlier["slot"],
                "stage0_b_order": later["order"],
                "stage0_b_dispatch": later["dispatch"],
                "stage0_b_slot": later["slot"],
                "busy_overlap_ns": overlap,
                "envelope_gap_ns": later["device_start_ns"] - earlier["device_end_ns"],
            }
        )

    positive_pairs = [pair for pair in overlap_pairs if pair["busy_overlap_ns"] > 0]
    assigned_kernels = sum(len(kernels) for kernels in kernels_by_stage.values())
    agents = {
        agent["id"]["handle"]: {
            "logical_node_id": agent.get("logical_node_id"),
            "logical_node_type_id": agent.get("logical_node_type_id"),
            "gpu_id": agent.get("gpu_id"),
            "name": agent.get("name"),
            "product_name": agent.get("product_name"),
        }
        for agent in tool["agents"]
        if agent.get("type") == 2
    }

    output = {
        "method": "HIP API correlation IDs inside GGML_SCHED_TRACE host enqueue ranges mapped to rocprofv3 kernel completion intervals",
        "trace": str(args.trace),
        "log": str(args.log),
        "gpu_agents": agents,
        "stage0_backend": stage0_backend,
        "stage1_backend": stage1_backend,
        "meta_stage_intervals": len(stages),
        "meta_stage_intervals_with_kernels": len(stage_rows),
        "kernel_records": len(kernel_records),
        "assigned_kernel_records": assigned_kernels,
        "assignment_fraction": assigned_kernels / len(kernel_records) if kernel_records else 0.0,
        "stage0_to_previous_stage1_pairs": len(overlap_pairs),
        "positive_busy_overlap_pairs": len(positive_pairs),
        "total_busy_overlap_ns": sum(pair["busy_overlap_ns"] for pair in overlap_pairs),
        "max_busy_overlap_ns": max((pair["busy_overlap_ns"] for pair in overlap_pairs), default=0),
        "overlap_proven": bool(positive_pairs),
        "overlap_pairs": overlap_pairs,
        "stages": [
            {key: value for key, value in row.items() if key != "busy_intervals"}
            for row in stage_rows
        ],
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        "HYBRID_DEVICE_TIMELINE_OK",
        f"assigned={assigned_kernels}/{len(kernel_records)}",
        f"pairs={len(overlap_pairs)}",
        f"positive={len(positive_pairs)}",
        f"max_overlap_ns={output['max_busy_overlap_ns']}",
        f"overlap_proven={int(output['overlap_proven'])}",
    )


if __name__ == "__main__":
    main()
