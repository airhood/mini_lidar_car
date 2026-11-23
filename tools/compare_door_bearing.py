"""Compare fixed and static-door guidance on identical Unity doorway trials."""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

from inspect_direction_transfer import CASE, read_batch
from prepare_door_bearing import door_rows, target


def audit_runtime(root, layouts, exit_follow=False):
    samples = 0
    maximum_error = 0.0
    for path in sorted((root / "recordings").glob("*.csv")):
        match = CASE.match(path.name)
        layout = layouts[int(match[4]), int(match[3])]
        doors = door_rows(layout[2])
        exit_z = layout[4][0] if exit_follow else None
        with path.open(newline="") as source:
            for row in csv.DictReader(source):
                x, z = float(row["x_m"]), float(row["z_m"])
                # Geometry in logs is rounded to millimetres. Ignore gate-switch ties.
                if any(abs(z - (back + 0.28)) < 0.002 for _, back in doors):
                    continue
                wx, wz = target(x, z, doors, exit_z)
                error = math.hypot(wx - float(row["waypoint_x_m"]), wz - float(row["waypoint_z_m"]))
                maximum_error = max(maximum_error, error)
                samples += 1
    if not samples or maximum_error > 0.006:
        raise ValueError(f"Runtime bearing does not match static geometry ({maximum_error})")
    return {"scans": samples, "max_world_vector_difference": maximum_error}


def evaluate(teacher_path, fixed_path, bearing_path, exit_follow=False):
    teacher_layouts, teacher = read_batch(teacher_path)
    layouts, fixed = read_batch(fixed_path, require_fixed_direction=not exit_follow)
    bearing_layouts, bearing = read_batch(bearing_path, require_fixed_direction=False)
    if len(layouts) != 12 or layouts != bearing_layouts or teacher_layouts.keys() - layouts.keys():
        raise ValueError("Expected the same twelve trials for both policy conditions")
    eligible = {key for key, row in teacher.items() if row["outcome"] == "completed"}
    for key in eligible:
        if teacher_layouts[key] != layouts[key]:
            raise ValueError(f"Changed geometry: {key}")
        for cases in (fixed, bearing):
            if (cases[key]["positions"][0] != teacher[key]["positions"][0]
                    or cases[key]["start_heading"] != teacher[key]["start_heading"]):
                raise ValueError(f"Different starting pose: {key}")

    result = {"requested": len(layouts), "teacher_completed": len(eligible),
              "runtime_bearing_verification": audit_runtime(bearing_path, layouts, exit_follow)}
    labels = ("door_only", "door_and_exit") if exit_follow else ("fixed", "bearing")
    gates = {key: door_rows(layouts[key][2]) for key in eligible}
    for label, cases in zip(labels, (fixed, bearing)):
        result[label] = {
            "outcomes": dict(Counter(cases[key]["outcome"] for key in eligible)),
            "doors_cleared": [
                sum(cases[key]["furthest_z_m"] >= gates[key][i][1] + 0.28 for key in eligible)
                for i in range(2)
            ],
            "completed_after_second_door": sum(
                cases[key]["outcome"] == "completed" and cases[key]["furthest_z_m"] >= gates[key][1][1] + 0.28
                for key in eligible
            ),
            "cases": {f"{seed}/{task}": {k: v for k, v in cases[(seed, task)].items() if k != "positions"}
                      for seed, task in sorted(eligible)},
        }
    return result, (layouts, teacher, fixed, bearing, eligible)


def plot(path, cohorts, policy_names=("Fixed direction", "Next-door bearing")):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, 2, figsize=(9, 6), layout="constrained")
    for ax, (cohort, data) in zip(axes, cohorts.items()):
        layouts, teacher, fixed, bearing, eligible = data
        key = min(eligible, key=lambda key: (
            bearing[key]["outcome"] != "completed", fixed[key]["outcome"] == "completed", key))
        family, area, walls, boxes, (exit_z, half_width) = layouts[key]
        for rect in walls:
            ax.add_patch(Rectangle(rect[:2], *rect[2:], color="0.28", lw=0))
        for rect in boxes:
            ax.add_patch(Rectangle(rect[:2], *rect[2:], color="0.65", lw=0))
        ax.plot([-half_width, half_width], [exit_z, exit_z], ":", color="0.4", label="Exit")
        for label, cases in (("Teacher", teacher), (policy_names[0], fixed), (policy_names[1], bearing)):
            x, z = zip(*cases[key]["positions"])
            line, = ax.plot(x, z, "--" if label == "Teacher" else "-", label=label, lw=1.6)
            ax.scatter(x[-1], z[-1], marker="x", c=[line.get_color()], s=28)
        x, z, width, length = area
        ax.set(aspect="equal", xlim=(x - .2, x + width + .2), ylim=(z - .2, z + length + .2),
               xlabel="x (m)", ylabel="z (m)", title=f"{cohort} | {family}\n{key[0]}/{key[1]}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=4, fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("previous", type=Path)
    parser.add_argument("--door-only", type=Path, help="existing door-only evaluation for the exit comparison")
    args = parser.parse_args()
    p, old = args.experiment, args.previous
    if args.door_only is None:
        previous, previous_traces = evaluate(old / "teacher-heldout", old / "short", p / "bearing-previous")
        new, new_traces = evaluate(p / "teacher-new", p / "fixed-new", p / "bearing-new")
        labels = ("fixed", "bearing")
        policy_names = ("Fixed direction", "Next-door bearing")
    else:
        door = args.door_only
        previous, previous_traces = evaluate(old / "teacher-heldout", door / "bearing-previous",
                                             p / "exit-previous", True)
        new, new_traces = evaluate(door / "teacher-new", door / "bearing-new", p / "exit-new", True)
        labels = ("door_only", "door_and_exit")
        policy_names = ("Door guidance", "Door and exit guidance")
    (p / "summary.json").write_text(json.dumps({"previous": previous, "new": new}, indent=2) + "\n")
    for name, result in (("Previous", previous), ("New", new)):
        for arm in labels:
            print(name, arm, result["teacher_completed"], result[arm]["outcomes"], result[arm]["doors_cleared"])
    plot(p / "comparison-trajectories.png", {"Previous": previous_traces, "New": new_traces}, policy_names)


if __name__ == "__main__":
    main()
