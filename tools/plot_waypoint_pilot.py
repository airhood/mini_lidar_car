"""Overlay held-out Unity trajectories on the actual hall geometry."""

import argparse
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from summarize_waypoint_pilot import CASE

GEOMETRY = re.compile(
    r"^Hall layout (\d+)/(\d+) (hall-[a-z]+) doors=\d+: [^|]+\| ([^|]+)\| (.+)$",
    re.MULTILINE,
)
MODELS = {
    "teacher-heldout": "Teacher",
    "final-goal": "Final goal",
    "waypoint": "Waypoint",
    "waypoint-mixed": "Waypoint + yaw training",
    "waypoint-blind": "Waypoint without LiDAR",
}


def track_paths(folder):
    tracks = {}
    for path in sorted((folder / "recordings").glob("*.csv")):
        _, task, seed, condition = CASE.match(path.name).groups()
        key = f"{seed}/{task}/{'clutter' if '-clutter' in path.name else 'clear'}/{condition}"
        tracks[key] = path
    return tracks


def rectangles(text):
    return [tuple(map(float, rect.split(","))) for rect in text.split(";") if rect.strip()]


def choose_cases(results):
    # Show one case that separates goal inputs and one of the shortest remaining waypoint runs.
    yaw_cases = [key for key in sorted(results["teacher-heldout"]["cases"])
                 if key.endswith("/clutter/yaw")]

    def completed(arm, key):
        return results[arm]["cases"][key]["outcome"] == "completed"

    strongest = max(
        yaw_cases,
        key=lambda key: (completed("waypoint", key) + completed("waypoint-mixed", key)
                         - 2 * completed("final-goal", key),
                         completed("waypoint-mixed", key) - completed("waypoint", key)),
    )
    remaining = [key for key in yaw_cases if key.split("/")[1] != strongest.split("/")[1]]
    failure = min(
        remaining,
        key=lambda key: (completed("waypoint-mixed", key), completed("waypoint", key),
                         results["waypoint-mixed"]["cases"][key]["travelled_m"]),
    )
    return [strongest.replace("/yaw", "/nominal"), strongest,
            failure.replace("/yaw", "/nominal"), failure]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    results = json.loads((root / "summary.json").read_text())
    models = {arm: name for arm, name in MODELS.items() if arm in results}
    tracks = {arm: track_paths(root / arm) for arm in models}
    halls = {(int(seed), int(task)): (family, rectangles(boxes), rectangles(walls))
             for seed, task, family, boxes, walls in
             GEOMETRY.findall((root / "teacher-heldout" / "unity.log").read_text())}

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for ax, key in zip(axes.flat, choose_cases(results)):
        seed, task, condition, variant = key.split("/")
        family, boxes, walls = halls[int(seed), int(task)]
        for x, z, width, height in walls:
            ax.add_patch(Rectangle((x, z), width, height, color="0.25", lw=0))
        for x, z, width, height in boxes:
            ax.add_patch(Rectangle((x, z), width, height, color="0.70", lw=0))

        traces = []
        for arm, name in models.items():
            path = tracks[arm][key]
            xy = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(3, 4), ndmin=2)
            linestyle = "--" if arm == "waypoint-blind" else "-"
            line, = ax.plot(*xy.T, lw=1.6, label=name, linestyle=linestyle)
            ax.scatter(*xy[-1], marker="x", c=[line.get_color()], s=32)
            if arm == "teacher-heldout":
                ax.scatter(*xy[0], label="Start", c="black", s=16, marker="o")
            traces.append(xy)
        points = np.concatenate(traces)
        low, high = points.min(axis=0) - 0.8, points.max(axis=0) + 0.8
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), aspect="equal",
               xlabel="x (m)", ylabel="z (m)")
        tilt = "nominal start" if variant == "nominal" else "yaw-disturbed start"
        ax.set_title(f"{family} | seed {int(seed)} / task {int(task)} | {tilt}", fontsize=10)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=5, fontsize=8)
    fig.savefig(root / "measured-trajectories.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
