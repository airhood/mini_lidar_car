"""Summarize held-out closed-loop runs, including first-box and recovery checks."""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re
from statistics import median

CASE = re.compile(
    r"^(completed|timeout|collision)-.*-task(\d+)-seed(\d+)-(nominal|yaw)-"
)
ARMS = ("teacher-heldout", "final-goal", "waypoint", "waypoint-mixed")


def measure(path):
    match = CASE.match(path.name)
    if not match:
        raise ValueError(f"Unfinished or unrecognized recording: {path.name}")
    outcome, task, seed, condition = match.groups()
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    first = rows[0]
    start_x = float(first["x_m"])
    start_z = float(first["z_m"])
    start_time = float(first["time_s"])
    recovery_seconds = None
    for row in rows:
        elapsed = float(row["time_s"]) - start_time
        if elapsed > 10 or int(row["collision_count"]):
            break
        x, z = float(row["x_m"]), float(row["z_m"])
        dx, dz = float(row["waypoint_x_m"]) - x, float(row["waypoint_z_m"]) - z
        error = (math.degrees(math.atan2(dx, dz)) + float(row["heading_deg"]) + 180) % 360 - 180
        if math.hypot(x - start_x, z - start_z) >= 1.0 and abs(error) <= 25:
            recovery_seconds = round(elapsed, 3)
            break
    distance = sum(
        math.hypot(float(second["x_m"]) - float(first["x_m"]),
                   float(second["z_m"]) - float(first["z_m"]))
        for first, second in zip(rows, rows[1:])
    )
    key = f"{seed}/{task}/{'clutter' if '-clutter' in path.name else 'clear'}/{condition}"
    return key, {
        "outcome": outcome,
        "travelled_m": round(distance, 3),
        "collision_before_1_m": outcome == "collision" and distance < 1.0,
        "has_checkpoint": first["has_box_checkpoint"] == "1",
        "passes_checkpoint": rows[-1]["passed_box_checkpoint"] == "1",
        "realigns_and_advances_within_10s": recovery_seconds is not None,
        "recovery_seconds": recovery_seconds,
    }


def report(cases):
    counts = Counter(run["outcome"] for run in cases)
    recovery = [run["recovery_seconds"] for run in cases if run["recovery_seconds"] is not None]
    return {
        "runs": len(cases),
        "outcomes": counts,
        "collision_before_1_m": sum(run["collision_before_1_m"] for run in cases),
        "with_box_checkpoint": sum(run["has_checkpoint"] for run in cases),
        "passes_box_checkpoint": sum(run["passes_checkpoint"] for run in cases),
        "realigns_and_advances_within_10s": len(recovery),
        "median_recovery_seconds_among_realigned": median(recovery) if recovery else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--blind-control", action="store_true")
    args = parser.parse_args()
    results = {}
    reference = None
    arms = (*ARMS, "waypoint-blind") if args.blind_control else ARMS
    for arm in arms:
        cases = {}
        for path in sorted((args.directory / arm / "recordings").glob("*.csv")):
            key, measurement = measure(path)
            if key in cases:
                raise ValueError(f"Repeated case: {key}")
            cases[key] = measurement
        checkpoints = {key: run["has_checkpoint"] for key, run in cases.items()}
        if arm == ARMS[0]:
            reference = checkpoints
            if len(reference) != 48:
                raise ValueError(f"Expected 48 teacher benchmarks, found {len(reference)}")
        elif checkpoints != reference:
            raise ValueError(f"{arm} benchmark cases differ from the teacher")
        results[arm] = {
            "nominal": report([run for key, run in cases.items() if key.endswith("/nominal")]),
            "yaw": report([run for key, run in cases.items() if key.endswith("/yaw")]),
            "cases": cases,
        }
    output = args.directory / "summary.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    for arm, result in results.items():
        print(arm, result["nominal"], result["yaw"])


if __name__ == "__main__":
    main()
