"""Verify geometry and score the matched Unity side-swap driving trials."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re

CASE = re.compile(r"^(completed|collision|timeout)-.*-swap(left|right)-task00-seed(\d+)-")
GEOMETRY = re.compile(r"^Side swap (\d+)/(left|right): ([^|]+)\| ([^|]+)\| exit_z=([^\n]+)$", re.M)


def load_batch(root):
    cases = {}
    layouts = {}
    for seed, side, area, box, exit_z in GEOMETRY.findall((root / "unity.log").read_text()):
        key = int(seed), side
        if key in layouts:
            raise ValueError(f"Repeated layout {key} in {root}")
        layouts[key] = tuple(map(float, area.split(","))), tuple(map(float, box.split(","))), float(exit_z)
    for path in sorted((root / "recordings").glob("*.csv")):
        match = CASE.match(path.name)
        if match is None:
            raise ValueError(f"Unexpected or unfinished recording {path}")
        outcome, side, seed = match.groups()
        key = int(seed), side
        if key in cases:
            raise ValueError(f"Repeated recording {key} in {root}")
        with path.open(newline="") as source:
            samples = list(csv.DictReader(source))
        _, (_, nose, _, length), _ = layouts[key]
        at_box = next((row for row in samples if float(row["z_m"]) >= nose + length * 0.5), None)
        lateral = float(at_box["x_m"]) if at_box else None
        cases[key] = {
            "outcome": outcome,
            "passed_box": samples[-1]["passed_box_checkpoint"] == "1",
            "correct_side_at_box": lateral is not None and (lateral > 0 if side == "left" else lateral < 0),
            "x_at_box_m": lateral,
            "samples": samples,
        }
    if cases.keys() != layouts.keys() or not cases:
        raise ValueError(f"Layout and recording keys differ in {root}")
    return cases, layouts


def audit_teacher(cases, layouts):
    for seed in sorted({seed for seed, _ in layouts}):
        left, right = (seed, "left"), (seed, "right")
        area, left_box, exit_z = layouts[left]
        other_area, right_box, other_exit_z = layouts[right]
        if (area != other_area or exit_z != other_exit_z or left_box[1:] != right_box[1:]
                or abs(left_box[0] + left_box[2] + right_box[0]) > 0.002
                or abs(left_box[0] - area[0]) > 0.002
                or abs(right_box[0] + right_box[2] - (area[0] + area[2])) > 0.002):
            raise ValueError(f"Expected mirror boxes flush with the walls: {seed}")
        first_samples = [cases[key]["samples"][0] for key in (left, right)]
        for field in ("x_m", "z_m", "heading_deg", "waypoint_x_m", "waypoint_z_m", "intent_id"):
            if abs(float(first_samples[0][field]) - float(first_samples[1][field])) > 0.0001:
                raise ValueError(f"Pair begins with unequal {field}: {seed}")
        for key in (left, right):
            case = cases[key]
            if not all((case["outcome"] == "completed", case["passed_box"], case["correct_side_at_box"])):
                raise ValueError(f"Teacher did not take the expected opposite bypass: {key}")
            for row in case["samples"]:
                dx = float(row["waypoint_x_m"]) - float(row["x_m"])
                dz = float(row["waypoint_z_m"]) - float(row["z_m"])
                if abs(dx) > 0.00001 or abs(dz - 1) > 0.00001:
                    raise ValueError(f"Direction was not constant in world coordinates: {key}")


def score(cases):
    seeds = {seed for seed, _ in cases}
    return {
        "outcomes": dict(Counter(case["outcome"] for case in cases.values())),
        "passed_box": sum(case["passed_box"] for case in cases.values()),
        "correct_side_at_box": sum(case["correct_side_at_box"] for case in cases.values()),
        "both_runs_completed": sum(all(cases[seed, side]["outcome"] == "completed"
                                       for side in ("left", "right")) for seed in seeds),
        "per_side": {side: dict(Counter(case["outcome"] for (_, case_side), case in cases.items()
                                        if side == case_side)) for side in ("left", "right")},
        "cases": {f"{seed}/{side}": {key: value for key, value in case.items() if key != "samples"}
                  for (seed, side), case in sorted(cases.items())},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--teachers-only", action="store_true")
    args = parser.parse_args()
    root = args.directory
    train, train_layouts = load_batch(root / "teacher")
    heldout, layouts = load_batch(root / "teacher-heldout")
    if set(train) & set(heldout) or len(train) != 56 or len(heldout) != 20:
        raise ValueError("Expected 28 development pairs and 10 disjoint evaluation pairs")
    audit_teacher(train, train_layouts)
    audit_teacher(heldout, layouts)
    results = {"teacher-heldout": score(heldout)}
    if not args.teachers_only:
        for arm in ("lidar", "blind"):
            cases, arm_layouts = load_batch(root / arm)
            if arm_layouts != layouts:
                raise ValueError(f"{arm} did not use the teacher's layouts")
            results[arm] = score(cases)
    (root / ("teacher-audit.json" if args.teachers_only else "summary.json")).write_text(
        json.dumps(results, indent=2) + "\n")
    for arm, result in results.items():
        print(arm, {key: value for key, value in result.items() if key != "cases"})


if __name__ == "__main__":
    main()
