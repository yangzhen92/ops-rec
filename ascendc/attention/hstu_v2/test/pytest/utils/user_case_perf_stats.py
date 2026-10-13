#!/usr/bin/env python3
"""Summarize current planned HSTU user-case profiles."""

import argparse
import glob
import json
from pathlib import Path

import pandas as pd


MODES = ("beta", "current_deterministic", "current_nondeterministic")
DIRECTIONS = ("forward", "backward")
PRE_OP_PATTERNS = {
    "af": ("BuildHstuMaskArbitraryFunc",),
    "packed_block_sparse": ("PackedBlockSparse",),
    "create_attn_metadata": ("CreateAttnMetadata",),
}


def load_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_profile(profile_dir):
    files = glob.glob(str(profile_dir / "**/op_summary_*.csv"), recursive=True)
    if not files:
        raise RuntimeError(f"no op_summary CSV under {profile_dir}")
    frames = [pd.read_csv(path, low_memory=False) for path in files]
    frame = pd.concat(frames, ignore_index=True)
    frame["Task Start Time(us)"] = pd.to_numeric(frame["Task Start Time(us)"], errors="coerce")
    frame["Task Duration(us)"] = pd.to_numeric(frame["Task Duration(us)"], errors="coerce")
    return frame.dropna(subset=["Task Start Time(us)", "Task Duration(us)"]).sort_values("Task Start Time(us)")


def load_durations(frame, op_name):
    selected = frame.loc[frame["Op Name"] == op_name].copy()
    selected["Task Start Time(us)"] = pd.to_numeric(selected["Task Start Time(us)"], errors="coerce")
    selected["Task Duration(us)"] = pd.to_numeric(selected["Task Duration(us)"], errors="coerce")
    selected = selected.dropna(subset=["Task Start Time(us)", "Task Duration(us)"])
    values = selected.sort_values("Task Start Time(us)")["Task Duration(us)"].tolist()
    return values


def classify_pre_op(name):
    for category, patterns in PRE_OP_PATTERNS.items():
        if any(pattern in name for pattern in patterns):
            return category
    return None


def measured_pre_ops(frame, op_name, records, warmup, repeats):
    """Attribute plan-side device tasks to the following HSTU launch."""
    events = frame.sort_values("Task Start Time(us)").to_dict("records")
    launches = []
    pending = {name: 0.0 for name in PRE_OP_PATTERNS}
    for event in events:
        name = str(event["Op Name"])
        category = classify_pre_op(name)
        if category is not None:
            pending[category] += float(event["Task Duration(us)"])
        if name == op_name:
            launches.append(pending)
            pending = {key: 0.0 for key in PRE_OP_PATTERNS}
    expected = len(records) * (warmup + repeats)
    if len(launches) != expected:
        raise RuntimeError(f"expected {expected} {op_name} launches, got {len(launches)}")
    rows = []
    per_case = warmup + repeats
    for index, record in enumerate(records):
        measured = launches[index * per_case + warmup : (index + 1) * per_case]
        for category in PRE_OP_PATTERNS:
            samples = [launch[category] for launch in measured]
            rows.append(
                {**record, "pre_op": category, "min_task_duration_us": min(samples), "samples_us": json.dumps(samples)}
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--xlsx", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    pre_rows = []
    for direction in DIRECTIONS:
        op_name = "HstuForwardV2" if direction == "forward" else "HstuBackwardV2"
        for mode in MODES:
            record_path = args.input_dir / "records" / f"{direction}_{mode}.jsonl"
            profile_dir = args.input_dir / "profiles" / f"{direction}_{mode}"
            if not record_path.exists() or not profile_dir.exists():
                continue
            records = load_rows(record_path)
            frame = load_profile(profile_dir)
            durations = load_durations(frame, op_name)
            expected = len(records) * (args.warmup + args.repeats)
            if len(durations) != expected:
                raise RuntimeError(f"{direction}/{mode}: expected {expected} {op_name} tasks, got {len(durations)}")
            per_case = args.warmup + args.repeats
            for index, record in enumerate(records):
                samples = durations[index * per_case + args.warmup : (index + 1) * per_case]
                rows.append({**record, "min_task_duration_us": min(samples), "samples_us": json.dumps(samples)})
            if mode != "beta":
                pre_rows.extend(measured_pre_ops(frame, op_name, records, args.warmup, args.repeats))
    result = pd.DataFrame(rows)
    current = result[result["mode"] != "beta"]
    pivot = current.pivot_table(
        index=["direction", "case_id", "pipeline"], columns="mode", values="min_task_duration_us"
    ).reset_index()
    pivot["nondet_vs_det_pct"] = (pivot["current_deterministic"] / pivot["current_nondeterministic"] - 1) * 100
    metadata = current.drop_duplicates(["direction", "case_id", "pipeline"]).drop(
        columns=["mode", "min_task_duration_us", "samples_us"]
    )
    comparison = metadata.merge(pivot, on=["direction", "case_id", "pipeline"])
    unsupported = []
    for path in sorted((args.input_dir / "records").glob("*.unsupported.jsonl")):
        unsupported.extend(load_rows(path))
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.csv, index=False)
    with pd.ExcelWriter(args.xlsx, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="raw", index=False)
        comparison.to_excel(writer, sheet_name="comparison", index=False)
        pd.DataFrame(pre_rows).to_excel(writer, sheet_name="pre_ops", index=False)
        pd.DataFrame(unsupported).to_excel(writer, sheet_name="unsupported", index=False)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
