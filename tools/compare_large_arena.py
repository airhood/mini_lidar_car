"""Compare training coverage and temporal architecture on held-out large maps."""

import argparse
from collections import Counter
import json
from pathlib import Path

from inspect_direction_transfer import read_batch

NAMES = {"teacher": "Teacher", "previous": "Previous CNN",
         "cnn": "CNN + large maps", "temporal": "CNN–GRU + large maps",
         "short": "CNN | 0.3 s", "long": "CNN | 1.5 s"}


def plot(root, layouts, arms, eligible):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    families = sorted({layout[0] for layout in layouts.values()})
    first, second = tuple(arms)[-2:]
    fig, axes = plt.subplots(1, len(families), figsize=(12, 6), layout="constrained")
    for ax, family in zip(axes, families):
        keys = [key for key in sorted(eligible) if layouts[key][0] == family]
        if not keys:
            ax.set_axis_off()
            continue
        key = min(keys, key=lambda key: (
            arms[first][key]["outcome"] == arms[second][key]["outcome"],
            arms["previous"][key]["outcome"] == "completed", key))
        _, area, walls, boxes, gate = layouts[key]
        for rect in walls:
            ax.add_patch(Rectangle(rect[:2], *rect[2:], color="0.28", lw=0))
        for rect in boxes:
            ax.add_patch(Rectangle(rect[:2], *rect[2:], color="0.66", lw=0))
        gate_z, half_width = gate
        ax.plot([-half_width, half_width], [gate_z, gate_z], ":", color="0.4", label="Exit")
        for name, cases in arms.items():
            x, z = zip(*cases[key]["positions"])
            line, = ax.plot(x, z, "--" if name == "teacher" else "-", label=NAMES[name], lw=1.5)
            ax.scatter(x[-1], z[-1], marker="x", c=[line.get_color()], s=26)
        x, z, width, length = area
        ax.set(aspect="equal", xlim=(x - .3, x + width + .3),
               ylim=(z - .3, z + length + .3), title=f"{family}\n{key[0]}/{key[1]}",
               xlabel="x (m)", ylabel="z (m)")
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="outside lower center", ncols=5, fontsize=8)
            break
    fig.savefig(root / "heldout-trajectories.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--doorway-history", action="store_true")
    args = parser.parse_args()
    compared_arms = ("short", "long") if args.doorway_history else ("cnn", "temporal")
    root = args.experiment
    development, development_runs = read_batch(root / "teacher-development")
    teacher_layouts, teacher = read_batch(root / "teacher-heldout")
    layouts, previous = read_batch(root / "previous")
    if len(layouts) != 12 or teacher_layouts.keys() - layouts.keys() or development.keys() & layouts.keys():
        raise ValueError("Evaluation needs twelve layouts disjoint from development")
    for key, layout in teacher_layouts.items():
        if layouts[key] != layout:
            raise ValueError(f"Different teacher and evaluation layout: {key}")
    eligible = {key for key, row in teacher.items() if row["outcome"] == "completed"}
    if not eligible:
        raise ValueError("No teacher completed maps")
    arms = {"teacher": teacher, "previous": previous}
    for name in compared_arms:
        arm_layouts, cases = read_batch(root / name)
        if arm_layouts != layouts:
            raise ValueError(f"Different layouts for {name}")
        arms[name] = cases
    summary = {"requested": len(layouts), "teacher_completed": len(eligible),
               "development_outcomes": dict(Counter(case["outcome"] for case in development_runs.values())),
               "by_arm": {}}
    for arm, cases in arms.items():
        for key in eligible:
            if (cases[key]["positions"][0] != teacher[key]["positions"][0]
                    or cases[key]["start_heading"] != teacher[key]["start_heading"]):
                raise ValueError(f"Different initial pose for {arm}/{key}")
        if arm == "teacher":
            continue
        summary["by_arm"][arm] = {
            "outcomes": dict(Counter(cases[key]["outcome"] for key in eligible)),
            "by_family": {family: dict(Counter(cases[key]["outcome"] for key in eligible
                                              if layouts[key][0] == family))
                          for family in sorted({layout[0] for layout in layouts.values()})},
            "cases": {f"{seed}/{task}": {k: v for k, v in cases[(seed, task)].items() if k != "positions"}
                      for seed, task in sorted(eligible)},
        }
        if args.doorway_history:
            door_edges = {key: sorted({z + length for _, z, _, length in layouts[key][2][4:]})
                          for key in eligible}
            if any(len(edges) != 2 for edges in door_edges.values()):
                raise ValueError("Expected two interior door rows")
            door_counts = [sum(cases[key]["furthest_z_m"] >= door_edges[key][door] + 0.28
                               for key in eligible) for door in range(2)]
            summary["by_arm"][arm]["doors_cleared"] = door_counts
            print(arm, "door passages", door_counts)
        print(arm, summary["by_arm"][arm]["outcomes"], summary["by_arm"][arm]["by_family"])
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(root, layouts, arms, eligible)


if __name__ == "__main__":
    main()
