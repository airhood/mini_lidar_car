"""Compare independently trained early wall-side policies on unchanged cases."""

import argparse
from collections import Counter
import json
from pathlib import Path

from inspect_direction_transfer import read_batch


def measure(cases):
    return {
        "outcomes": dict(Counter(case["outcome"] for case in cases.values())),
        "first_box_cleared": sum(case["boxes_cleared"] >= 1 for case in cases.values()),
        "second_box_cleared": sum(case["boxes_cleared"] >= 2 for case in cases.values()),
        "collisions_between_first_and_second": sum(
            case["outcome"] == "collision" and case["boxes_cleared"] == 1
            for case in cases.values()),
        "cases": {f"{seed}/{task}": {field: value for field, value in case.items() if field != "positions"}
                  for (seed, task), case in sorted(cases.items())},
    }


def compare(root, seed, teacher_path, baseline_path, previous_cases):
    reference, teacher = read_batch(teacher_path)
    layouts, baseline = read_batch(baseline_path)
    desired = {key: layout for key, layout in layouts.items() if layout[0] == "wall-side"}
    if len(desired) != 8:
        raise ValueError(f"Expected eight wall-side cases in {seed}")
    eligible = {key for key in desired if key in teacher and teacher[key]["outcome"] == "completed"}
    if not eligible:
        raise ValueError(f"No eligible cases for {seed}")
    results = {}
    for arm in ("baseline", "normal", "policy"):
        arm_layouts, cases = (desired, baseline) if arm == "baseline" else read_batch(root / f"{arm}-{seed}")
        if arm_layouts != desired:
            raise ValueError(f"Different wall-side layouts in {arm}/{seed}")
        for key in eligible:
            if (reference[key] != arm_layouts[key]
                    or cases[key]["positions"][0] != teacher[key]["positions"][0]
                    or cases[key]["start_heading"] != teacher[key]["start_heading"]):
                raise ValueError(f"Different evaluation start or geometry in {arm}/{key}")
        scored = {key: cases[key] for key in eligible}
        results[arm] = measure(scored)
        previous_cases[arm].update(scored)
    print(seed, {a: {k: v for k, v in m.items() if k != "cases"} for a, m in results.items()})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("experiments_root", type=Path)
    args = parser.parse_args()
    path = args.experiments_root
    pooled = {name: {} for name in ("baseline", "normal", "policy")}
    first = compare(args.experiment, 4200, path / "direction-stress-pilot/teacher",
                    path / "direction-stress-recovery/normal-heldout", pooled)
    second = compare(args.experiment, 4500, path / "stress-recovery-validation/teacher-new",
                     path / "stress-recovery-validation/candidate-new", pooled)
    results = {arm: measure(cases) for arm, cases in pooled.items()}
    (args.experiment / "comparison-summary.json").write_text(json.dumps({
        "by_cohort": {"4200": first, "4500": second},
        "pooled": results,
    }, indent=2) + "\n")
    print("Pooled", {a: {k: v for k, v in m.items() if k != "cases"} for a, m in results.items()})


if __name__ == "__main__":
    main()
