"""Select paired stationary restarts from recorded teacher and policy poses."""

import argparse
import csv
import json
import math
from pathlib import Path

from inspect_direction_transfer import CASE, read_batch


def load_poses(folder, families):
    poses = {}
    for path in sorted(folder.glob("*.csv")):
        match = CASE.match(path.name)
        if match is None or match[2] not in families:
            continue
        key = int(match[4]), int(match[3])
        if key in poses:
            raise ValueError(f"Repeated recording for {key}")
        with path.open(newline="") as source:
            poses[key] = [tuple(float(row[field]) for field in
                                ("x_m", "z_m", "heading_deg", "time_s"))
                          for row in csv.DictReader(source) if row["collision_count"] == "0"]
    return poses


def clearance(position, rect):
    x, z = position[:2]
    left, front, width, length = rect
    return math.hypot(max(left - x, 0.0, x - left - width),
                      max(front - z, 0.0, z - front - length))


def heading_difference(first, second):
    return abs((first - second + 180) % 360 - 180)


def choose_pair(teacher, policy, layout, early_wall_severity=None):
    family, _, walls, boxes, (exit_z, _) = layout
    passed_index = int(family == "wall-side" and early_wall_severity is None)
    _, front, _, length = boxes[passed_index]
    min_z = front + length + 0.30
    max_z = {
        "staggered": boxes[1][1] - 0.08,
        "angled-entry": exit_z - 0.25,
    }.get(family, boxes[-1][1] + boxes[-1][3])
    max_heading = 110 if family == "staggered" or early_wall_severity is not None else 170
    if early_wall_severity is not None:
        max_z = boxes[1][1] - 0.08

    def valid(pose):
        return min_z <= pose[1] <= max_z and min(
            clearance(pose, rect) for rect in (*walls, *boxes)) >= 0.43

    normal = [pose for pose in teacher if valid(pose)]
    candidates = []
    for pose in policy:
        if not normal or not valid(pose) or abs(pose[2]) > max_heading:
            continue
        reference = min(normal, key=lambda row: abs(row[1] - pose[1]))
        longitudinal = abs(reference[1] - pose[1])
        if longitudinal > 0.05:
            continue
        offset = abs(reference[0] - pose[0])
        angle = heading_difference(reference[2], pose[2])
        if offset >= 0.12 or angle >= 8:
            if early_wall_severity is None:
                score = offset + angle / 65 - longitudinal
            else:
                if offset > 0.65 or angle > 60:
                    continue
                severity = max(offset / 0.65, angle / 60)
                score = -abs(severity - early_wall_severity) - longitudinal
            candidates.append((score, reference, pose, offset, angle))
    return max(candidates, default=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("teacher", type=Path, help="teacher development batch directory")
    parser.add_argument("policy", type=Path, help="policy development batch directory")
    parser.add_argument("output", type=Path, help="new recovery-start CSV")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--stress", action="store_true", help="target angled-entry and wall-side states")
    modes.add_argument("--early-wall", action="store_true", help="target mild and moderate deviations after the first box")
    args = parser.parse_args()
    families = {"angled-entry", "wall-side"} if args.stress else {"staggered"}
    expected = 24 if args.stress else 20
    if args.early_wall:
        families = {"wall-side"}
        expected = 16
    teacher_layouts, teacher_cases = read_batch(args.teacher)
    policy_layouts, cases = read_batch(args.policy)
    normal = load_poses(args.teacher / "recordings", families)
    policy = load_poses(args.policy / "recordings", families)
    if not (len(policy_layouts) == expected and policy_layouts.keys() == policy.keys()):
        raise ValueError(f"Expected all {expected} policy development rollouts")
    selected = {}
    rejected = []
    for key, layout in sorted(policy_layouts.items()):
        if key not in teacher_cases or teacher_cases[key]["outcome"] != "completed":
            rejected.append({"seed": key[0], "task": key[1], "reason": "teacher did not complete"})
            continue
        if layout != teacher_layouts[key]:
            raise ValueError(f"Different teacher geometry for {key}")
        target = None
        if args.early_wall:
            # Use both target severities on the mirrored variants.
            target = 0.30 if (key[0] + key[1]) % 2 == 0 else 0.70
        pair = choose_pair(normal[key], policy[key], layout, target)
        if pair is None:
            rejected.append({"seed": key[0], "task": key[1], "reason": "no matched displaced pose"})
            continue
        _, normal_pose, policy_pose, offset, angle = pair
        selected[key] = {
            "family": layout[0],
            **({"severity_target": target} if args.early_wall else {}),
            "normal": normal_pose,
            "policy": policy_pose,
            "lateral_offset_m": round(offset, 3),
            "heading_difference_deg": round(angle, 2),
            "rollout_outcome": cases[key]["outcome"],
        }
    if not selected:
        raise ValueError("No safe, measurably off-track start pairs")
    with args.output.open("x", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(("seed", "task", "source", "x_m", "z_m", "heading_deg"))
        for (seed, task), pair in selected.items():
            for source in ("normal", "policy"):
                writer.writerow((seed, task, source, *pair[source][:3]))
    (args.output.parent / "start-selection.json").write_text(json.dumps({
        "pairs": {f"{seed}/{task}": pair for (seed, task), pair in selected.items()},
        "rejected": rejected,
    }, indent=2) + "\n")
    print(f"Selected {len(selected)} paired starts from {len(cases)} rollouts")


if __name__ == "__main__":
    main()
