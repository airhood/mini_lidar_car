"""Plot measured seed-1129 Unity trajectories with the actual walls and obstacles."""

import argparse
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import numpy as np

from audit_layout_geometry import audit_layouts, coordinates, graph_wall_edges
from summarize_policy_evaluation import RECORDING

RUNS = {"Teacher": "teacher-1129", "v8": "v8-1129", "v8 without previous cmd": "v8-no-previous-1129"}
CASES = (
    ("network", 1, "clear", "Corridor right turn, clear"),
    ("network", 0, "box-approach", "Corridor with single obstacle"),
    ("hall-partitions", 0, "clear", "Partitioned hall, clear"),
    ("hall-chain", 2, "clutter09", "Linked rooms with nine obstacles"),
)
GOAL_CASES = (
    ("hall-partitions", 0, "clear", "Partitioned hall, clear"),
    ("hall-chain", 2, "clear", "Linked rooms, clear"),
    ("hall-atrium", 1, "clutter12", "Atrium with twelve obstacles"),
    ("hall-chain", 2, "clutter09", "Linked rooms with nine obstacles"),
)


def load_tracks(folders):
    tracks = {}
    for name, folder in folders.items():
        trajectories = {}
        for path in sorted((folder / "recordings").glob("*.csv")):
            match = RECORDING.fullmatch(path.name)
            if match is None or int(match["seed"]) != 1129:
                continue
            key = match["family"], int(match["task"]), match["placement"]
            rows = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 1, 2, 3, 4), ndmin=2)
            trajectories[key] = rows, match["outcome"]
        tracks[name] = trajectories
    return tracks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--floorplans", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--goal-pilot", type=Path, help="compare the two goal pilot policies")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    folders = {name: args.evaluation / folder for name, folder in RUNS.items()}
    cases = CASES
    if args.goal_pilot:
        folders = {"Teacher": folders["Teacher"],
                   "No goal": args.goal_pilot / "no-goal-1129",
                   "Relative goal": args.goal_pilot / "relative-goal-1129"}
        cases = GOAL_CASES
    tracks = load_tracks(folders)
    graph_layouts, _ = audit_layouts(args.floorplans)
    _, _, _, graph_walls, graph_box, _, _ = next(layout for layout in graph_layouts if layout[0] == "1129/graph")
    hall_layouts = {}
    source = (args.evaluation / "teacher-1129/unity.log").read_text()
    pattern = r"^Hall layout 1129/(\d+) (hall-[a-z]+) doors=\d+: [^|]+\| ([^|]+)\| (.+)$"
    for mission, family, boxes, walls in re.findall(pattern, source, re.M):
        hall_layouts[(family, int(mission))] = coordinates(walls), coordinates(boxes)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    metrics = {}
    for ax, (family, task, placement, title) in zip(axes.flat, cases):
        key = family, task, placement
        teacher = tracks["Teacher"][key][0]
        reference = teacher[:, 3:5]
        endpoint = reference[-1]
        panel = {}
        traces = []
        for model, episodes in tracks.items():
            rows, outcome = episodes[key]
            xy = rows[:, 3:5]
            time = rows[:, 0] - rows[0, 0]
            line, = ax.plot(*xy.T, label=model, lw=1.6)
            ax.scatter(*xy[-1], marker="x", c=[line.get_color()], s=36)
            commanded_forward = rows[:, 1:3].mean(axis=1)
            panel[model] = {
                "outcome": outcome,
                "travelled_m": round(float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()), 3),
                "final_distance_to_teacher_endpoint_m": round(float(np.linalg.norm(xy[-1] - endpoint)), 3),
                "closest_distance_to_teacher_endpoint_m": round(float(np.linalg.norm(xy - endpoint, axis=1).min()), 3),
                "mean_forward_command_first_three_seconds": round(float(commanded_forward[time < 3].mean()), 4),
            }
            traces.append(xy)
        metrics[f"{family}/{task:02d}/{placement}"] = panel
        if family == "network":
            ax.add_collection(LineCollection(graph_wall_edges(graph_walls), colors="0.45", linewidths=0.8, zorder=1))
            boxes = graph_box if placement != "clear" else []
        else:
            walls, boxes = hall_layouts[(family, task)]
            for x, z, w, h in walls:
                ax.add_patch(Rectangle((x, z), w, h, color="0.25", lw=0, zorder=1))
            if placement == "clear":
                boxes = []
        for x, z, w, h in boxes:
            ax.add_patch(Rectangle((x, z), w, h, color="0.55", lw=0, zorder=1))
        ax.scatter(*reference[0], marker="o", c="black", s=14, label="Start")
        ax.scatter(*endpoint, marker="*", c="black", s=72, label="Teacher finish")
        points = np.concatenate(traces)
        lower, upper = points.min(axis=0) - 0.9, points.max(axis=0) + 0.9
        ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), aspect="equal",
               xlabel="x (m)", ylabel="z (m)")
        outcomes = " | ".join(f"{name}: {record['outcome']}" for name, record in panel.items() if name != "Teacher")
        ax.set_title(f"{title} (task {task:02d})\n{outcomes}", fontsize=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=5, fontsize=8)
    fig.savefig(args.output / "measured-trajectories.png", dpi=160)
    plt.close(fig)
    (args.output / "trajectory-metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
