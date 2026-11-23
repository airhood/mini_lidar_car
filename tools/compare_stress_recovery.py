"""Compare equally sampled recovery policies on the same harder Unity courses."""

import argparse
from collections import Counter
import json
from pathlib import Path

from inspect_direction_stress import FAMILIES
from inspect_direction_transfer import read_batch

LABELS = {"baseline": "Previous policy", "normal": "Normal restarts", "policy": "Policy-pose restarts"}


def plot(root, layouts, teacher, arms, eligible):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, 2, figsize=(8, 7), layout="constrained")
    for ax, family in zip(axes, ("angled-entry", "wall-side")):
        keys = [key for key in eligible if layouts[key][0] == family]
        key = min(keys, key=lambda key: (
            arms["policy"][key]["outcome"] == arms["normal"][key]["outcome"],
            arms["policy"][key]["outcome"] == "completed",
            key,
        ))
        _, area, walls, boxes, (exit_z, half_width) = layouts[key]
        for rect in walls:
            ax.add_patch(Rectangle(rect[:2], *rect[2:], color="0.30", lw=0))
        for rect in boxes:
            ax.add_patch(Rectangle(rect[:2], *rect[2:], color="0.65", lw=0))
        ax.plot([-half_width, half_width], [exit_z, exit_z], ":", color="0.4", label="Exit")
        for name, cases in (("Teacher", teacher), *((LABELS[arm], cases) for arm, cases in arms.items())):
            x, z = zip(*cases[key]["positions"])
            line, = ax.plot(x, z, "--" if name == "Teacher" else "-", label=name)
            ax.scatter(x[-1], z[-1], color=line.get_color(), marker="x")
        x, z, width, height = area
        ax.set(aspect="equal", xlim=(x - .17, x + width + .17), ylim=(z - .17, z + height + .17),
               xlabel="x (m)", ylabel="z (m)", title=f"{family} | {key[0]}/{key[1]}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, fontsize=8)
    fig.savefig(root / "comparison-trajectories.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("benchmark", type=Path)
    args = parser.parse_args()
    root = args.experiment
    teacher_layouts, teacher = read_batch(args.benchmark / "teacher")
    layouts, baseline = read_batch(args.benchmark / "recovery-lidar")
    arms = {"baseline": baseline}
    for arm in ("normal", "policy"):
        arm_layouts, cases = read_batch(root / f"{arm}-heldout")
        if arm_layouts != layouts:
            raise ValueError(f"Mismatched evaluation layouts for {arm}")
        arms[arm] = cases
    eligible = {key for key, case in teacher.items() if case["outcome"] == "completed"}
    if len(eligible) != 22 or len(layouts) != 24:
        raise ValueError("Expected 22 physically verified cases and 24 planned cases")
    for key in eligible:
        if layouts[key] != teacher_layouts[key]:
            raise ValueError(f"Mismatched teacher layout for {key}")
        for arm, cases in arms.items():
            if (cases[key]["positions"][0] != teacher[key]["positions"][0]
                    or cases[key]["start_heading"] != teacher[key]["start_heading"]):
                raise ValueError(f"Mismatched starting pose for {arm}/{key}")

    summary = {}
    for arm, cases in arms.items():
        counts = {}
        for family in FAMILIES:
            keys = [key for key in eligible if layouts[key][0] == family]
            counts[family] = dict(Counter(cases[key]["outcome"] for key in keys))
        summary[arm] = {
            "by_family": counts,
            "outcomes": dict(Counter(cases[key]["outcome"] for key in eligible)),
            "second_box_cleared": sum(cases[key]["boxes_cleared"] >= 2 for key in eligible),
            "cases": {f"{seed}/{task}": {k: v for k, v in cases[(seed, task)].items() if k != "positions"}
                      for seed, task in sorted(eligible)},
        }
        print(arm, {k: v for k, v in summary[arm].items() if k != "cases"})
    (root / "comparison-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(root, layouts, teacher, arms, eligible)


if __name__ == "__main__":
    main()
