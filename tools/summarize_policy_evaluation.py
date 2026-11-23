"""Summarize one or more Unity evaluation batches against the same scenarios."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

import numpy as np

RECORDING = re.compile(
    r"(?P<outcome>completed|stopped|collision|timeout|interrupted)-"
    r"(?:teacher-v12|policy-[a-z0-9-]+)-(?P<family>network|rooms|hall-[a-z]+)-"
    r"(?P<intent>follow|left|right)-(?P<placement>clear|blocked|box-(?:approach|exit)|clutter\d+)-"
    r"task(?P<task>\d+)-seed(?P<seed>\d+)-\d{8}-\d{6}-[a-f0-9]+\.csv"
)
BATCH = re.compile(r"^Teacher batch finished: (\d+) saved, (\d+) skipped, (\d+) failed\.$", re.M)


def read_run(folder):
    log = (folder / "unity.log").read_text()
    batch = BATCH.search(log)
    if not batch:
        raise ValueError(f"{folder}: missing batch completion summary")

    cases = {}
    by_environment = defaultdict(Counter)
    by_condition = defaultdict(Counter)
    for path in sorted((folder / "recordings").glob("*.csv")):
        match = RECORDING.fullmatch(path.name)
        if not match:
            raise ValueError(f"Unrecognized CSV: {path.name}")
        outcome, family, intent, placement, task, seed = match.groups()
        key = f"{seed}/{task}/{family}/{intent}/{placement}"
        if key in cases:
            raise ValueError(f"Duplicate scenario in {folder.name}: {key}")
        groups = "hall" if family.startswith("hall-") else family
        role = "clutter" if placement.startswith("clutter") else placement
        by_environment[groups][outcome] += 1
        by_condition[role][outcome] += 1
        rows = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 3, 4), ndmin=2)
        cases[key] = {
            "outcome": outcome,
            "recorded_seconds": round(float(rows[-1, 0] - rows[0, 0]), 3) if len(rows) else 0,
            "distance_m": round(float(np.linalg.norm(np.diff(rows[:, 1:], axis=0), axis=1).sum()), 3),
        }
    counts = Counter(case["outcome"] for case in cases.values())
    successes, skipped, failures = map(int, batch.groups())
    if successes != counts["completed"] + counts["stopped"] or len(cases) != successes + failures:
        raise ValueError(f"{folder}: CSV outcomes disagree with the Unity summary")
    return {
        "attempted": successes + skipped + failures,
        "unavailable_geometries": skipped,
        "outcomes": dict(counts),
        "by_environment": {group: dict(values) for group, values in sorted(by_environment.items())},
        "by_condition": {group: dict(values) for group, values in sorted(by_condition.items())},
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+", help="each run has a Unity log and recordings directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = {path.name: read_run(path) for path in args.runs}
    if len(runs) != len(args.runs):
        parser.error("run names must be distinct")
    first = next(iter(runs.values()))
    if any(run["cases"].keys() != first["cases"].keys() for run in runs.values()):
        parser.error("batch scenarios do not match")
    args.output.write_text(json.dumps(runs, indent=2) + "\n")
    print(json.dumps({name: {key: value for key, value in run.items() if key != "cases"}
                      for name, run in runs.items()}, indent=2))


if __name__ == "__main__":
    main()
