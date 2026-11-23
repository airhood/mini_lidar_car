"""Compare policy and teacher on held-out obstacle geometry stress cases."""

import argparse
from collections import Counter
import json
from pathlib import Path

from inspect_direction_transfer import read_batch

FAMILIES = ("short-gap", "angled-entry", "wall-side")


def plot(root, layouts, teacher, policy, eligible):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, 3, figsize=(11, 6), layout="constrained")
    for ax, family in zip(axes, FAMILIES):
        keys = [key for key in eligible if layouts[key][0] == family]
        if not keys:
            ax.set_axis_off()
            continue
        key = min(keys, key=lambda key: (policy[key]["outcome"] == "completed",
                                         policy[key]["boxes_cleared"] == 0,
                                         policy[key]["furthest_z_m"], key))
        _, area, walls, boxes, (exit_z, half_width) = layouts[key]
        for rect in walls:
            ax.add_patch(Rectangle(rect[:2], *rect[2:], color="0.3", lw=0))
        for rect in boxes:
            ax.add_patch(Rectangle(rect[:2], *rect[2:], color="0.64", lw=0))
        ax.plot([-half_width, half_width], [exit_z, exit_z], ":", color="0.4", label="Exit")
        for name, case, style in (("Teacher", teacher[key], "--"), ("Policy", policy[key], "-")):
            x, z = zip(*case["positions"])
            line, = ax.plot(x, z, style, label=name)
            ax.scatter(x[-1], z[-1], marker="x", color=line.get_color(), s=28)
        x, z, width, height = area
        ax.set(aspect="equal", xlim=(x - .17, x + width + .17), ylim=(z - .17, z + height + .17),
               title=f"{family}\n{key[0]}/{key[1]} | {policy[key]['outcome']}",
               xlabel="x (m)", ylabel="z (m)")
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="outside lower center", ncols=3)
            break
    fig.savefig(root / "measured-trajectories.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    teacher_layouts, teacher = read_batch(root / "teacher")
    layouts, policy = read_batch(root / "recovery-lidar")
    if len(layouts) != 24 or (set(teacher_layouts) - set(layouts)):
        raise ValueError("Expected the twenty-four requested stress cases")
    for key, layout in teacher_layouts.items():
        if (layout != layouts[key]
                or teacher[key]["positions"][0] != policy[key]["positions"][0]
                or teacher[key]["start_heading"] != policy[key]["start_heading"]):
            raise ValueError(f"Policy and teacher configurations differ for {key}")
    eligible = {key for key, case in teacher.items() if case["outcome"] == "completed"}
    results = {}
    for family in FAMILIES:
        keys = [key for key in sorted(layouts) if layouts[key][0] == family]
        feasible = [key for key in keys if key in eligible]
        results[family] = {
            "requested": len(keys),
            "teacher_outcomes": dict(Counter(teacher[key]["outcome"] if key in teacher else "skipped"
                                             for key in keys)),
            "policy_outcomes_on_teacher_completed": dict(Counter(policy[key]["outcome"] for key in feasible)),
            "first_box_cleared": sum(policy[key]["boxes_cleared"] > 0 for key in feasible),
            "collision_after_first_box": sum(policy[key]["outcome"] == "collision"
                                              and policy[key]["boxes_cleared"] > 0 for key in feasible),
            "cases": {f"{seed}/{task}": {
                "teacher": teacher[seed, task]["outcome"] if (seed, task) in teacher else "skipped",
                **{k: v for k, v in policy[seed, task].items() if k != "positions"},
            } for seed, task in keys},
        }
        print(family, {k: v for k, v in results[family].items() if k != "cases"})
    (root / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    plot(root, layouts, teacher, policy, eligible)


if __name__ == "__main__":
    main()
