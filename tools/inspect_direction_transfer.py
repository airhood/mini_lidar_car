"""Score and plot the same Unity cases across direction-conditioned policies."""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re

CASE = re.compile(r"^(completed|collision|timeout)-.*-direction-([a-z-]+)-follow-task(\d+)-seed(\d+)-")
LAYOUT = re.compile(
    r"^Direction transfer (\d+)/(\d+) ([a-z-]+): ([^|]+)\| ([^|]+)\| ([^|]*)\| exit=([^\n]+)$",
    re.MULTILINE,
)
FAMILIES = ("freestanding", "staggered", "corridor-hall", "clear")


def rects(value):
    return [tuple(map(float, rect.split(","))) for rect in value.strip().split(";") if rect]


def read_batch(root, require_fixed_direction=True):
    layouts = {}
    for seed, task, family, area, walls, boxes, exit_gate in LAYOUT.findall(
            (root / "unity.log").read_text()):
        key = int(seed), int(task)
        if key in layouts:
            raise ValueError(f"Duplicate layout {key} in {root}")
        layouts[key] = (family, rects(area)[0], rects(walls), rects(boxes),
                        tuple(map(float, exit_gate.split(","))))
    cases = {}
    for path in sorted((root / "recordings").glob("*.csv")):
        match = CASE.match(path.name)
        if match is None:
            raise ValueError(f"Incomplete or unrecognized run: {path.name}")
        outcome, family, task, seed = match.groups()
        key = int(seed), int(task)
        if key in cases or layouts[key][0] != family:
            raise ValueError(f"Mismatched or repeated run: {path.name}")
        with path.open(newline="") as source:
            rows = list(csv.DictReader(source))
        for row in rows:
            dx = float(row["waypoint_x_m"]) - float(row["x_m"])
            dz = float(row["waypoint_z_m"]) - float(row["z_m"])
            if require_fixed_direction:
                if abs(dx) > 0.00001 or abs(dz - 1) > 0.00001:
                    raise ValueError(f"Unexpected travel direction in {path.name}")
            elif abs(math.hypot(dx, dz) - 1) > 0.00001:
                raise ValueError(f"Expected a unit travel direction in {path.name}")
        positions = [(float(row["x_m"]), float(row["z_m"])) for row in rows]
        furthest = max(z for _, z in positions)
        boxes = layouts[key][3]
        cleared = sum(furthest >= z + length + 0.28 for _, z, _, length in boxes)
        tail_start = float(rows[-1]["time_s"]) - 10
        tail = [(row, position) for row, position in zip(rows, positions)
                if float(row["time_s"]) >= tail_start]
        tail_distance = sum(math.dist(a[1], b[1]) for a, b in zip(tail, tail[1:]))
        tail_forward = sum((float(row["left_cmd"]) + float(row["right_cmd"])) * 0.5
                           for row, _ in tail) / len(tail)
        cases[key] = {
            "outcome": outcome,
            "boxes_cleared": cleared,
            "furthest_z_m": round(furthest, 3),
            "last_10s_distance_m": round(tail_distance, 3),
            "last_10s_mean_forward_command": round(tail_forward, 4),
            "positions": positions,
            "start_heading": float(rows[0]["heading_deg"]),
        }
    if cases.keys() != layouts.keys():
        raise ValueError(f"Layout and recording keys differ in {root}")
    return layouts, cases


def summarize(layouts, cases):
    summary = {}
    for family in FAMILIES:
        keys = [key for key in layouts if layouts[key][0] == family]
        counts = Counter(cases[key]["outcome"] for key in keys)
        summary[family] = {
            "attempted": len(keys),
            "outcomes": dict(counts),
            "first_box_cleared": sum(cases[key]["boxes_cleared"] >= 1 for key in keys),
            "all_boxes_cleared": sum(
                bool(layouts[key][3]) and cases[key]["boxes_cleared"] == len(layouts[key][3])
                for key in keys),
        }
    return summary


def plot(root, layouts, policies):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    frozen = policies["frozen-lidar"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 9), layout="constrained")
    for ax, family in zip(axes.flat, FAMILIES):
        keys = [key for key in sorted(layouts) if layouts[key][0] == family
                and policies["teacher-heldout"][key]["outcome"] == "completed"]
        if not keys:
            ax.set_axis_off()
            continue
        key = min(keys, key=lambda key: (
            frozen[key]["outcome"] == "completed",
            family == "staggered" and frozen[key]["boxes_cleared"] == 0,
            frozen[key]["furthest_z_m"],
        ))
        _, area, walls, boxes, gate = layouts[key]
        for wall in walls:
            ax.add_patch(Rectangle(wall[:2], *wall[2:], color="0.30", lw=0))
        for box in boxes:
            ax.add_patch(Rectangle(box[:2], *box[2:], color="0.65", lw=0))
        gate_z, half_width = gate
        ax.plot([-half_width, half_width], [gate_z, gate_z], ":", color="0.4", label="Exit gate")
        for name, cases in policies.items():
            x, z = zip(*cases[key]["positions"])
            line, = ax.plot(x, z, label=name.replace("-", " "),
                            linestyle="--" if name == "teacher-heldout" else "-")
            ax.scatter(x[-1], z[-1], marker="x", color=line.get_color(), s=28)
        x, z, width, length = area
        ax.set(aspect="equal", xlim=(x - .17, x + width + .17),
               ylim=(z - .17, z + length + .17), xlabel="x (m)", ylabel="z (m)",
               title=f"{family} | seed {key[0]}, task {key[1]}")
    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="outside lower center", ncols=4, fontsize=8)
            break
    fig.savefig(root / "measured-trajectories.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--policy", action="append", default=[])
    args = parser.parse_args()
    root = args.directory
    layouts, teacher = read_batch(root / "teacher-heldout")
    if len(layouts) != 35:
        raise ValueError(f"Expected all 35 held-out cases, found {len(layouts)}")
    policies = {"teacher-heldout": teacher}
    for arm in ("frozen-lidar", *args.policy):
        matching_layouts, cases = read_batch(root / arm)
        if matching_layouts != layouts:
            raise ValueError(f"{arm} saw different layouts")
        for key, case in cases.items():
            if (teacher[key]["positions"][0] != case["positions"][0]
                    or teacher[key]["start_heading"] != case["start_heading"]):
                raise ValueError(f"{arm} started at a different pose in {key}")
        policies[arm] = cases
    results = {
        name: {
            "by_family": summarize(layouts, cases),
            "cases": {f"{seed}/{task}": {"family": layouts[seed, task][0],
                                        **{k: v for k, v in case.items() if k != "positions"}}
                      for (seed, task), case in sorted(cases.items())},
        }
        for name, cases in policies.items()
    }
    (root / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    for name, result in results.items():
        print(name, result["by_family"])
    plot(root, layouts, policies)


if __name__ == "__main__":
    main()
