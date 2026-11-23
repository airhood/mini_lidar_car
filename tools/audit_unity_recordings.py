"""Audit completed Unity v8 recordings without opening or changing the scene."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

import numpy as np

NAME = re.compile(
    r"(?P<outcome>completed|stopped|collision|timeout|interrupted)-teacher-v8-"
    r"(?P<family>network|rooms)-(?P<intent>follow|left|right)-task(?P<task>\d+)"
    r"-seed(?P<seed>\d+)-\d{8}-\d{6}-[\da-f]+\.csv"
)
ANGLES = np.deg2rad(np.arange(360))
COSINE, SINE = np.cos(ANGLES), np.sin(ANGLES)


def summary(values):
    if not values:
        return None
    return {key: round(float(value), 4) for key, value in zip(
        ("min", "median", "max"), np.percentile(values, (0, 50, 100))
    )}


def read_episode(path):
    with path.open() as file:
        header = file.readline().rstrip().split(",")
        expected = ["time_s", "left_cmd", "right_cmd", "x_m", "z_m", "heading_deg",
                    "collision_count", "intent_id", *(f"range_{i}_m" for i in range(360))]
        if header != expected:
            raise ValueError("unexpected CSV schema")
        rows = np.loadtxt(file, delimiter=",", ndmin=2)
    if len(rows) < 2 or len(header) != rows.shape[1]:
        raise ValueError("missing observation or incomplete CSV row")
    if not np.all(np.isfinite(rows[:, :8])):
        raise ValueError("invalid metadata")
    return rows


def inspect(directory):
    outcomes = Counter()
    intents = Counter()
    files_by_family = Counter()
    invalid = []
    near_stops = []
    final_gaps, final_speeds = [], []
    scan_count = bad_spacing = collisions = 0
    pairs = defaultdict(dict)

    for path in sorted(directory.glob("*.csv")):
        match = NAME.fullmatch(path.name)
        if match is None:
            continue
        label = match.groupdict()
        outcomes[label["outcome"]] += 1
        files_by_family[label["family"]] += 1
        try:
            rows = read_episode(path)
        except (ValueError, OSError) as error:
            invalid.append({"file": path.name, "error": str(error)})
            continue
        intervals = np.diff(rows[:, 0])
        scan_count += len(rows)
        bad_spacing += int(np.count_nonzero(np.abs(intervals - 0.1) > 0.002))
        collisions += int(np.any(rows[:, 6] > 0))
        intents.update(dict(zip(*np.unique(rows[:, 7].astype(int), return_counts=True))))
        task = int(label["task"])
        if task < 8 and label["outcome"] == "completed":
            first = rows[0]
            pairs[(label["seed"], task // 2)][task % 2] = (
                label["intent"], first[3:6], np.minimum(first[8:], 6.0), rows[:, 1:3],
            )

        if label["outcome"] != "stopped" or intervals[-1] <= 0:
            continue
        ranges = rows[-1, 8:]
        valid = np.isfinite(ranges)
        x = np.where(valid, ranges, 0) * COSINE
        y = np.where(valid, ranges, 0) * SINE
        ahead = valid & (x > 0) & (np.abs(y) <= 0.18)
        speed = float(np.linalg.norm(rows[-1, 3:5] - rows[-2, 3:5]) / intervals[-1])
        final_speeds.append(speed)
        if np.any(ahead):
            gap = float(x[ahead].min() - 0.20)
            final_gaps.append(gap)
            if gap < 0.15:
                near_stops.append({"file": path.name, "front_gap_m": round(gap, 4)})

    initial_scan_mae, split_times = [], []
    unmatched_pose = 0
    for pair in pairs.values():
        if not {0, 1}.issubset(pair) or pair[0][0] == pair[1][0]:
            continue
        a, b = pair[0], pair[1]
        position_delta = float(np.linalg.norm(a[1][:2] - b[1][:2]))
        heading_delta = abs((a[1][2] - b[1][2] + 180.0) % 360.0 - 180.0)
        if position_delta > 0.002 or heading_delta > 0.10:
            unmatched_pose += 1
            continue
        initial_scan_mae.append(float(np.abs(a[2] - b[2]).mean()))
        overlap = min(len(a[3]), len(b[3]))
        changed = np.flatnonzero(np.max(np.abs(a[3][:overlap] - b[3][:overlap]), axis=1) >= 0.10)
        if len(changed):
            split_times.append(float(changed[0]) * 0.1)

    return {
        "outcomes": dict(outcomes), "episodes_by_family": dict(files_by_family),
        "scans": scan_count, "scans_by_intent": {int(k): int(v) for k, v in intents.items()},
        "invalid_files": invalid, "irregular_intervals": bad_spacing, "contact_episodes": collisions,
        "last_front_clearance_m": summary(final_gaps), "last_interval_speed_m_s": summary(final_speeds),
        "stopped_episodes_with_front_clearance_below_0_15_m": near_stops,
        "paired_different_intents_same_pose": len(initial_scan_mae),
        "paired_different_intents_pose_mismatch": unmatched_pose,
        "paired_initial_scan_mae_m": summary(initial_scan_mae),
        "paired_command_divergence_time_s": summary(split_times),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.dumps(inspect(args.directory), indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
