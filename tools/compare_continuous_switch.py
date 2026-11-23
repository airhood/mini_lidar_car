"""Check a guidance switch during uninterrupted driving with the same learned weights."""

import argparse
import csv
from itertools import zip_longest
import json
from pathlib import Path

from compare_door_bearing import audit_runtime, evaluate, plot
from inspect_direction_transfer import CASE
from prepare_door_bearing import door_rows


def recordings(directory):
    result = {}
    for path in (directory / "recordings").glob("*.csv"):
        case = CASE.match(path.name)
        key = int(case[4]), int(case[3])
        if key in result:
            raise ValueError(f"Repeated continuous run: {key}")
        result[key] = path
    return result


def prefix(previous, switched, layouts, eligible):
    old, new = recordings(previous), recordings(switched)
    matched_runs = 0
    matched_samples = 0
    discrepancies = []
    columns = ("x_m", "z_m", "heading_deg", "left_cmd", "right_cmd", "waypoint_x_m", "waypoint_z_m")
    for key in sorted(eligible):
        _, back = door_rows(layouts[key][2])[-1]
        valid_frames = 0
        with old[key].open(newline="") as old_file, new[key].open(newline="") as new_file:
            for i, (a, b) in enumerate(zip_longest(csv.DictReader(old_file), csv.DictReader(new_file))):
                if a is None or b is None:
                    if a is not b and float((a or b)["z_m"]) <= back + 0.278:
                        discrepancies.append({"case": key, "scan": i, "reason": "ended before shared switch"})
                    break
                if max(float(a["z_m"]), float(b["z_m"])) > back + 0.278:
                    break
                if any(abs(float(a[name]) - float(b[name])) > 0.0001 for name in columns):
                    discrepancies.append({"case": key, "scan": i, "reason": "pre-switch numeric difference"})
                    break
                valid_frames += 1
        if not discrepancies or tuple(discrepancies[-1]["case"]) != key:
            matched_runs += 1
        matched_samples += valid_frames
    return {"matched_runs": matched_runs, "eligible_runs": len(eligible), "matched_scans": matched_samples,
            "differences_before_switch": discrepancies}


def compare(teachers, switched, previous):
    # evaluate() uses the ranking labels of the earlier experiment; the first arm
    # takes the forward switch and the second retains exit guidance here.
    raw, data = evaluate(teachers, switched, previous, exit_follow=True)
    layouts, teacher, forward, exit_angle, eligible = data
    summary = {
        "teacher_completed": raw["teacher_completed"],
        "forward_after_door": raw["door_only"],
        "exit_bearing_after_door": raw["door_and_exit"],
        "forward_bearing_audit": audit_runtime(switched, layouts),
        "pre_switch_comparison": prefix(previous, switched, layouts, eligible),
        "only_forward_completed": sum(forward[k]["outcome"] == "completed" and exit_angle[k]["outcome"] != "completed"
                                      for k in eligible),
        "only_exit_completed": sum(exit_angle[k]["outcome"] == "completed" and forward[k]["outcome"] != "completed"
                                   for k in eligible),
    }
    return summary, (layouts, teacher, exit_angle, forward, eligible)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("experiments", type=Path)
    args = parser.parse_args()
    p, root = args.output, args.experiments
    refs = root / "exit-bearing-pilot"
    previous, previous_rows = compare(root / "doorway-history-pilot/teacher-heldout",
                                     p / "forward-previous", refs / "exit-previous")
    new, new_rows = compare(root / "door-bearing-pilot/teacher-new",
                           p / "forward-new", refs / "exit-new")
    (p / "summary.json").write_text(json.dumps({"previous": previous, "new": new}, indent=2) + "\n")
    for tag, result in (("previous", previous), ("new", new)):
        print(tag, result["teacher_completed"], result["forward_after_door"]["outcomes"],
              result["exit_bearing_after_door"]["outcomes"], result["pre_switch_comparison"])
    plot(p / "comparison-trajectories.png", {"Previous": previous_rows, "New": new_rows},
         ("Exit bearing", "Forward after second door"))


if __name__ == "__main__":
    main()
