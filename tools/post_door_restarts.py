"""Select common post-door starts and compare guidance independently of training."""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

from compare_door_bearing import audit_runtime
from inspect_direction_transfer import CASE, read_batch
from prepare_door_bearing import door_rows
from prepare_mid_route_recovery import clearance, heading_difference

ARMS = {
    "door-model-fixed": False,
    "door-model-exit": True,
    "exit-model-fixed": False,
    "exit-model-exit": True,
}


def first_safe_pose(path, layout):
    _, _, walls, boxes, (exit_z, half_width) = layout
    _, back = door_rows(walls)[-1]
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            p = tuple(float(row[name]) for name in ("x_m", "z_m", "heading_deg"))
            if (row["collision_count"] == "0" and back + 0.42 <= p[1] <= back + 1.10
                    and p[1] < exit_z - 0.8 and abs(p[0]) <= half_width
                    and min(clearance(p, rect) for rect in (*walls, *boxes)) >= 0.41):
                return p
    return None


def recording_paths(folder):
    paths = {}
    for path in (folder / "recordings").glob("*.csv"):
        match = CASE.match(path.name)
        if match is None:
            raise ValueError(f"Invalid recording: {path}")
        key = int(match[4]), int(match[3])
        if key in paths:
            raise ValueError(f"Repeated recording for {key}")
        paths[key] = path
    return paths


def prepare(output, teachers, policies=()):
    candidates = {}
    for folder in policies:
        layouts, cases = read_batch(folder, require_fixed_direction=False)
        for key, path in recording_paths(folder).items():
            candidates.setdefault(key, []).append((layouts[key], path, folder.name, cases[key]["outcome"]))
    selected = {}
    excluded = {}
    for folder in teachers:
        layouts, cases = read_batch(folder)
        for key, path in sorted(recording_paths(folder).items()):
            if cases[key]["outcome"] != "completed":
                continue
            if key in selected or key in excluded:
                raise ValueError(f"Repeated reference course: {key}")
            layout = layouts[key]
            reference = first_safe_pose(path, layout)
            if reference is None:
                excluded[key] = "no eligible teacher start near second doorway"
                continue
            pose = reference
            details = {"source": "teacher"}
            if policies:
                options = []
                for actual_layout, actual_path, model, outcome in candidates.get(key, []):
                    if actual_layout != layout:
                        raise ValueError(f"Policy geometry differs for {key}")
                    actual = first_safe_pose(actual_path, layout)
                    if actual is not None:
                        lateral = abs(actual[0] - reference[0])
                        heading = heading_difference(actual[2], reference[2])
                        options.append((lateral + heading / 60, model, actual, lateral, heading, outcome))
                if not options:
                    excluded[key] = "neither policy supplied an eligible start"
                    continue
                _, model, pose, lateral, heading, outcome = max(options)
                details = {"source": model, "source_rollout_outcome": outcome, "teacher_pose": reference,
                           "lateral_offset_m": round(lateral, 3), "heading_difference_deg": round(heading, 2)}
            _, back = door_rows(layout[2])[-1]
            selected[key] = {"family": layout[0], "pose": pose,
                             "distance_past_door_m": round(pose[1] - back, 3), **details}
    if not selected:
        raise ValueError("No post-door starts found")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "starts.csv").open("x", newline="") as destination:
        writer = csv.writer(destination, lineterminator="\n")
        writer.writerow(("seed", "task", "source", "x_m", "z_m", "heading_deg"))
        for (seed, task), details in sorted(selected.items()):
            writer.writerow((seed, task, "policy" if policies else "normal", *details["pose"]))
    (output / "start-selection.json").write_text(json.dumps({
        "selected": {f"{seed}/{task}": details for (seed, task), details in sorted(selected.items())},
        "excluded": {f"{seed}/{task}": reason for (seed, task), reason in sorted(excluded.items())},
    }, indent=2) + "\n")
    print(f"Selected {len(selected)} paired starts and excluded {len(excluded)}")


def measure(cases, eligible):
    return {
        "outcomes": dict(Counter(cases[k]["outcome"] for k in eligible)),
        "cases": {f"{seed}/{task}": {name: value for name, value in cases[(seed, task)].items()
                                   if name != "positions"} for seed, task in sorted(eligible)},
    }


def verify_starts(requested, cases):
    for key, case in cases.items():
        first = case["positions"][0]
        x, z, heading = requested[key]
        if math.dist(first, (x, z)) > 0.001 or heading_difference(heading, case["start_heading"]) > 0.1:
            raise ValueError(f"Different initial pose for {key}")


def compare(root, teacher_pose_reference=None):
    with (root / "starts.csv").open(newline="") as source:
        expected = {(int(row["seed"]), int(row["task"])):
                    (float(row["x_m"]), float(row["z_m"]), float(row["heading_deg"]))
                    for row in csv.DictReader(source)}
    layouts, teacher = read_batch(root / "teacher")
    if layouts.keys() - expected.keys():
        raise ValueError("Unexpected teacher restarts")
    verify_starts(expected, teacher)
    eligible = {key for key, case in teacher.items() if case["outcome"] == "completed"}
    if not eligible:
        raise ValueError("No teacher completed restarts")
    results = {"requested": len(expected), "teacher_completed": len(eligible),
               "teacher_outcomes": dict(Counter(teacher[key]["outcome"] if key in teacher else "skipped"
                                                for key in expected)),
               "by_arm": {}, "paired_guidance_effect": {}}
    policy = {}
    for label, toward_exit in ARMS.items():
        batch = root / label
        arm_layouts, cases = read_batch(batch, require_fixed_direction=False)
        if arm_layouts.keys() != expected.keys() or any(arm_layouts[key] != value for key, value in layouts.items()):
            raise ValueError(f"Different layouts for {label}")
        verify_starts(expected, cases)
        results["by_arm"][label] = measure(cases, eligible)
        results["by_arm"][label]["guidance_audit"] = audit_runtime(batch, arm_layouts, exit_follow=toward_exit)
        policy[label] = cases
        print(label, results["by_arm"][label]["outcomes"])
    for name in ("door-model", "exit-model"):
        fixed, guided = policy[f"{name}-fixed"], policy[f"{name}-exit"]
        results["paired_guidance_effect"][name] = {
            "only_exit_completed": sum(guided[k]["outcome"] == "completed" and fixed[k]["outcome"] != "completed"
                                       for k in eligible),
            "only_fixed_completed": sum(fixed[k]["outcome"] == "completed" and guided[k]["outcome"] != "completed"
                                        for k in eligible),
        }
    if teacher_pose_reference is not None:
        reference = json.loads(teacher_pose_reference.read_text())
        results["matched_teacher_pose_comparison"] = {}
        for label in ARMS:
            reference_outcomes = reference["by_arm"][label]["cases"]
            results["matched_teacher_pose_comparison"][label] = dict(Counter(
                reference_outcomes[f"{seed}/{task}"]["outcome"] for seed, task in eligible))
    (root / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print("Teacher", results["teacher_outcomes"], "paired", results["paired_guidance_effect"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prepare-from", nargs=2, type=Path)
    parser.add_argument("--policy-from", nargs="+", type=Path, default=[])
    parser.add_argument("--teacher-pose-reference", type=Path)
    args = parser.parse_args()
    if args.prepare_from:
        prepare(args.output, args.prepare_from, args.policy_from)
    else:
        compare(args.output, args.teacher_pose_reference)


if __name__ == "__main__":
    main()
