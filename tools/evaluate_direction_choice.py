"""Score paired Unity runs where only the requested direction changes."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
from statistics import median

from inspect_direction_transfer import rects

SIDE = ("left", "right")
CASE = re.compile(
    r"^(completed|collision|timeout)-.*-direction-choice-(island|fork)-(left|right)-seed(\d+)-")
LAYOUT = re.compile(
    r"^Direction choice (\d+)/(island|fork)/(left|right): ([^|]+)\| ([^|]+)\| exit=([^\n]+)$",
    re.MULTILINE,
)
BEAMS = [f"range_{index}_m" for index in range(360)]
INITIAL_STATE = ["x_m", "z_m", "heading_deg", "intent_id", *BEAMS]


def crossing(rows, island):
    left, front, width, length = island
    plane = front + length / 2
    for (x0, z0), (x1, z1) in zip(rows, rows[1:]):
        if z0 < plane <= z1:
            cross_x = x0 + (x1 - x0) * (plane - z0) / (z1 - z0)
            if cross_x < left:
                return "left"
            if cross_x > left + width:
                return "right"
            return "through-island"
    return None


def read_batch(path, *, check_stationary_commands=False):
    geometry = {}
    for seed, family, side, area, island, exit_z in LAYOUT.findall((path / "unity.log").read_text()):
        key = int(seed), family, side
        if key in geometry:
            raise ValueError(f"Repeated layout: {key}")
        geometry[key] = rects(area)[0], rects(island)[0], float(exit_z)

    runs = {}
    for file in (path / "recordings").glob("*.csv"):
        match = CASE.match(file.name)
        if not match:
            raise ValueError(f"Unexpected or incomplete recording: {file}")
        outcome, family, side, seed = match.groups()
        key = int(seed), family, side
        if key in runs or key not in geometry:
            raise ValueError(f"Repeated or mismatched recording: {file}")
        with file.open(newline="") as source:
            rows = list(csv.DictReader(source))
        if len(rows) < 2:
            raise ValueError(f"Insufficient scan records: {file}")
        first = rows[0]
        if check_stationary_commands and any(float(first[f"{which}_cmd"]) != 0 for which in SIDE):
            raise ValueError(f"Expected stationary start: {file}")
        pose = [(float(row["x_m"]), float(row["z_m"])) for row in rows]
        direction = float(first["waypoint_x_m"]) - pose[0][0]
        turn = (float(rows[1]["left_cmd"]) - float(rows[1]["right_cmd"])) / 2
        runs[key] = {"outcome": outcome, "requested": side, "actual": crossing(pose, geometry[key][1]),
                     "first_turn": turn, "start": [first[column] for column in INITIAL_STATE],
                     "first_direction_x": direction, "positions": pose, "seconds": float(rows[-1]["time_s"]) -
                     float(first["time_s"])}
    if not runs or geometry.keys() != runs.keys():
        raise ValueError(f"Missing recordings or layouts in {path}")
    return geometry, runs


def counts(runs):
    return dict(Counter(
        "no_crossing" if run["actual"] is None else
        "requested_side" if run["actual"] == run["requested"] else
        "through_island" if run["actual"] == "through-island" else "other_side"
        for run in runs))


def render(path, keys, layout, runs):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(4, (len(keys) + 3) // 4, figsize=(13, 11), squeeze=False,
                             layout="constrained")
    for ax, (seed, family) in zip(axes.flat, keys):
        room, island, exit_z = layout[seed, family, "left"]
        ax.add_patch(Rectangle(island[:2], *island[2:], color="0.4"))
        ax.axhline(exit_z, linestyle=":", linewidth=0.7, color="0.55")
        for side, color in zip(SIDE, ("C0", "C1")):
            track = runs[seed, family, side]
            x, z = zip(*track["positions"])
            ax.plot(x, z, color=color, linewidth=1.3, label=f"{side.capitalize()} instruction")
            ax.plot(x[-1], z[-1], "x" if track["outcome"] == "collision" else ".", color=color)
        ax.set(xlim=(room[0], room[0] + room[2]), ylim=(-0.30, exit_z + 0.25),
               aspect="equal", title=f"{seed} · {family}")
    for ax in list(axes.flat)[len(keys):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2)
    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("x (m)")
    fig.supylabel("z (m)")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--teacher", type=Path, help="reuse an existing matched teacher evaluation")
    args = parser.parse_args()
    path = args.experiment
    geometry, teacher = read_batch(args.teacher or path / "teacher")
    policy_geometry, policy = read_batch(path / "policy", check_stationary_commands=True)
    if geometry != policy_geometry:
        raise ValueError("Teacher and policy layouts differed.")
    pairs = sorted({key[:2] for key in policy})
    eligible = []
    for seed, family in pairs:
        left, right = (seed, family, "left"), (seed, family, "right")
        if (geometry[left] != geometry[right] or policy[left]["start"] != policy[right]["start"]
                or teacher[left]["start"] != teacher[right]["start"]
                or policy[left]["start"] != teacher[left]["start"]):
            raise ValueError(f"Initial geometry, pose, or LiDAR differed: {seed}/{family}")
        if not policy[left]["first_direction_x"] < 0 < policy[right]["first_direction_x"]:
            raise ValueError(f"Initial instructions did not differ by side: {seed}/{family}")
        if all(teacher[key]["outcome"] == "completed" and teacher[key]["actual"] == key[-1]
               for key in (left, right)):
            eligible.append((seed, family))
    summaries = {}
    for name, selected in (("all", eligible),
                           ("island", [pair for pair in eligible if pair[1] == "island"]),
                           ("fork", [pair for pair in eligible if pair[1] == "fork"])):
        runs = [policy[(*pair, side)] for pair in selected for side in SIDE]
        summaries[name] = {
            "pairs": len(selected), "outcomes": dict(Counter(run["outcome"] for run in runs)),
            "crossing": counts(runs),
            "both_requested_sides": sum(all(policy[(*pair, side)]["actual"] == side for side in SIDE)
                                        for pair in selected),
            "both_completed_as_requested": sum(all(policy[(*pair, side)]["actual"] == side
                                                    and policy[(*pair, side)]["outcome"] == "completed"
                                                    for side in SIDE) for pair in selected),
            "same_side": sum(policy[(*pair, "left")]["actual"] is not None
                             and policy[(*pair, "left")]["actual"] == policy[(*pair, "right")]["actual"]
                             for pair in selected),
            "first_turn_both_correct_signs": sum(policy[(*pair, "left")]["first_turn"] < 0 <
                                                  policy[(*pair, "right")]["first_turn"] for pair in selected),
            "median_first_turn_right_minus_left": median(
                policy[(*pair, "right")]["first_turn"] - policy[(*pair, "left")]["first_turn"]
                for pair in selected) if selected else None,
            "first_turn_left_lower_than_right": sum(policy[(*pair, "left")]["first_turn"] <
                                                    policy[(*pair, "right")]["first_turn"] for pair in selected),
        }
    result = {"requested_pairs": len(pairs), "teacher_validated_pairs": len(eligible), "summary": summaries,
              "cases": {f"{seed}/{family}/{side}": {
                  arm: {key: value for key, value in batch[(seed, family, side)].items()
                        if key not in {"start", "positions"}} for arm, batch in (("teacher", teacher), ("policy", policy))
              } for seed, family in pairs for side in SIDE}}
    (path / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    if eligible:
        render(path / "trajectories.png", eligible, geometry, policy)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
