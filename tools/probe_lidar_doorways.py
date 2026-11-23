"""Detect wall openings from scan geometry and score them on recorded Unity runs."""

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from inspect_direction_transfer import CASE, LAYOUT, rects

BEAM_COLUMNS = [f"range_{i}_m" for i in range(360)]
BEAM_ANGLES = np.deg2rad(np.arange(360))
BEAM_DIRECTIONS = np.column_stack((-np.sin(BEAM_ANGLES), np.cos(BEAM_ANGLES)))
NORMAL_ANGLES = np.deg2rad(np.arange(0, 180, 2))
NORMALS = np.column_stack((np.sin(NORMAL_ANGLES), np.cos(NORMAL_ANGLES)))
OFFSET_STEP = 0.08
OFFSET_BINS = 161
ZERO_BIN = OFFSET_BINS // 2
MAX_RANGE = 6.0


def opening_candidates(ranges):
    """Return (center, plane normal, width) in the scanner's (right, forward) frame."""
    valid = np.isfinite(ranges) & (ranges >= 0.15) & (ranges <= MAX_RANGE)
    points = ranges[valid, None] * BEAM_DIRECTIONS[valid]
    if len(points) < 12:
        return []

    offsets = np.rint((NORMALS @ points.T) / OFFSET_STEP).astype(np.int32) + ZERO_BIN
    indices = offsets + OFFSET_BINS * np.arange(len(NORMALS))[:, None]
    votes = np.bincount(indices.ravel(), minlength=OFFSET_BINS * len(NORMALS))
    peaks = np.argpartition(votes, -35)[-35:]
    planes = []
    candidates = []

    for peak in sorted(peaks, key=lambda index: (-votes[index], index)):
        if votes[peak] < 9:
            break
        angle_bin, offset_bin = divmod(peak, OFFSET_BINS)
        normal = NORMALS[angle_bin]
        offset = (offset_bin - ZERO_BIN) * OFFSET_STEP
        seed = points[abs(points @ normal - offset) < 0.055]
        if len(seed) < 9:
            continue

        center = seed.mean(axis=0)
        centered = seed - center
        _, axes = np.linalg.eigh(centered.T @ centered)
        normal = axes[:, 0]
        if normal[1] < 0:
            normal = -normal
        offset = normal @ center
        if any(normal @ other > 0.996 and abs(offset - old) < 0.13 for other, old in planes):
            continue
        planes.append((normal, offset))

        tangent = np.array([normal[1], -normal[0]])
        support = np.sort(points[abs(points @ normal - offset) < 0.055] @ tangent)
        if len(support) < 12:
            continue
        gaps = np.diff(support)
        for gap in np.flatnonzero((gaps >= 1.40) & (gaps <= 3.60)):
            low, high = support[gap:gap + 2]
            left = support[(support >= low - 0.8) & (support <= low)]
            right = support[(support >= high) & (support <= high + 0.8)]
            if min(len(left), len(right)) < 3 or min(np.ptp(left), np.ptp(right)) < 0.15:
                continue

            midpoint = offset * normal + ((low + high) / 2) * tangent
            if not any(np.linalg.norm(midpoint - existing[0]) < 0.35 for existing in candidates):
                candidates.append((midpoint, normal, float(high - low)))
    return candidates


def scan_points(ranges):
    valid = np.isfinite(ranges) & (ranges >= 0.15) & (ranges <= MAX_RANGE)
    return np.where(valid, ranges, np.nan)[:, None] * BEAM_DIRECTIONS


def rotation(angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine], [sine, cosine]])


def scan_motion(points, previous):
    """Fit the current scan to the preceding one without any vehicle pose input."""
    left = np.roll(previous, 2, axis=0)
    right = np.roll(previous, -2, axis=0)
    tangent = right - left
    length = np.linalg.norm(tangent, axis=1)
    surface = (length > 0.025) & (length < 0.45)
    normals = np.full_like(previous, np.nan)
    normals[surface] = tangent[surface, ::-1] * [1, -1] / length[surface, None]
    current = points[np.isfinite(points[:, 0])]
    if len(current) < 24:
        return None

    angle = 0.0
    shift = np.zeros(2)
    beam_offsets = np.arange(-4, 5)
    for _ in range(4):
        moved = current @ rotation(angle).T
        transformed = moved + shift
        bearing = np.rad2deg(np.arctan2(-transformed[:, 0], transformed[:, 1]))
        indices = (np.rint(bearing).astype(int)[:, None] + beam_offsets) % 360
        refs = previous[indices]
        errors = np.sum((refs - transformed[:, None]) ** 2, axis=2)
        errors[~np.isfinite(errors) | ~surface[indices]] = np.inf
        nearest = errors.argmin(axis=1)
        active = errors[np.arange(len(current)), nearest] < 0.15 ** 2
        if active.sum() < 24:
            return None

        chosen = indices[np.arange(len(current)), nearest][active]
        normal = normals[chosen]
        perpendicular = moved[active, ::-1] * [-1, 1]
        jacobian = np.column_stack((normal, np.sum(perpendicular * normal, axis=1)))
        residual = np.sum((previous[chosen] - transformed[active]) * normal, axis=1)
        delta, _, _, singular = np.linalg.lstsq(jacobian, residual, rcond=None)
        if len(singular) < 3 or singular[-1] < 0.05 * singular[0]:
            return None
        shift += delta[:2]
        angle += delta[2]

    if np.linalg.norm(shift) > 0.09 or abs(angle) > np.deg2rad(8):
        return None
    return rotation(angle), shift


@dataclass
class OpeningTrack:
    center: np.ndarray
    normal: np.ndarray
    width: float
    seen: int = 1
    missed: int = 0
    passed: bool = False


class OpeningTracker:
    """Keep short-lived openings in the robot frame by matching adjacent scans."""

    def __init__(self):
        self.previous = None
        self.tracks = []

    def update(self, ranges, candidates):
        points = scan_points(ranges)
        if self.previous is not None:
            motion = scan_motion(points, self.previous)
            if motion is None:
                self.tracks.clear()
            else:
                orient, shift = motion
                for track in self.tracks:
                    track.center = orient.T @ (track.center - shift)
                    track.normal = orient.T @ track.normal
                    track.missed += 1
        self.previous = points

        available = list(self.tracks)
        for center, normal, width in candidates:
            compatible = [track for track in available
                          if abs(track.width - width) <= 0.38
                          and abs(track.normal @ normal) >= 0.97
                          and np.linalg.norm(track.center - center) <= 0.38]
            if compatible:
                track = min(compatible, key=lambda item: np.linalg.norm(item.center - center))
                available = [item for item in available if item is not track]
                track.center = 0.75 * track.center + 0.25 * center
                track.width = 0.75 * track.width + 0.25 * width
                if track.normal @ normal < 0:
                    normal = -normal
                track.normal = 0.75 * track.normal + 0.25 * normal
                track.normal /= np.linalg.norm(track.normal)
                track.seen += 1
                track.missed = 0
            elif normal @ center > 0.30 and center[1] > 0:
                self.tracks.append(OpeningTrack(center, normal, width))

        self.tracks = [track for track in self.tracks if track.missed <= (25 if track.seen >= 3 else 3)]
        events = []
        for track in self.tracks:
            tangent = track.normal[::-1] * [1, -1]
            if (track.seen >= 3 and not track.passed and track.normal @ track.center < -0.40
                    and abs(tangent @ track.center) < track.width / 2 - 0.15):
                track.passed = True
                events.append(track)
        ahead = [track for track in self.tracks if track.seen >= 3 and not track.passed
                 and track.normal @ track.center > -0.40]
        return min(ahead, key=lambda item: np.linalg.norm(item.center), default=None), events


def match_door(center, normal, width, doors):
    for index, (door_x, front, back, door_width) in enumerate(doors):
        if (abs(center[0] - door_x) <= 0.35
                and min(abs(center[1] - front), abs(center[1] - back)) <= 0.16
                and abs(width - door_width) <= 0.45 and abs(normal[1]) >= 0.94):
            return index
    return None


def score(path, doors):
    counts = Counter()
    tracker = OpeningTracker()
    reported = []
    passes = [{"ground_truth_time_s": None, "first_rear_observation_s": None} for _ in doors]
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            t = float(row["time_s"])
            position = np.array([float(row["x_m"]), float(row["z_m"])])
            heading = np.deg2rad(-float(row["heading_deg"]))
            cosine, sine = np.cos(heading), np.sin(heading)
            to_world = np.array([[cosine, sine], [-sine, cosine]])
            ranges = np.array([float(row[column]) for column in BEAM_COLUMNS])

            candidates = opening_candidates(ranges)
            tracked, events = tracker.update(ranges, candidates)
            for event in events:
                world = position + to_world @ event.center
                door = match_door(world, to_world @ event.normal, event.width, doors)
                reported.append({"time_s": t, "matched_door": door})
            matches = []
            for center, normal, width in candidates:
                world_center = position + to_world @ center
                door = match_door(world_center, to_world @ normal, width, doors)
                matches.append((center, door))
                counts["candidates"] += 1
                counts["candidates_matching_a_door"] += door is not None

            upcoming = next((i for i, (_, _, back, _) in enumerate(doors)
                             if position[1] <= back + 0.28), None)
            if upcoming is not None:
                front = doors[upcoming][1]
                tracked_door = None
                if tracked is not None:
                    tracked_door = match_door(position + to_world @ tracked.center,
                                              to_world @ tracked.normal, tracked.width, doors)
                if 0.30 <= front - position[1] <= 4.5:
                    counts["approach_scans"] += 1
                    counts["tracked_ahead_correct"] += tracked_door == upcoming
                    counts["tracked_ahead_wrong"] += tracked is not None and tracked_door != upcoming
                    counts["next_door_in_candidates"] += any(door == upcoming for _, door in matches)
                    ahead = [(float(np.linalg.norm(center)), door) for center, door in matches
                             if center[1] > 0.0]
                    if ahead:
                        nearest = min(ahead, key=lambda item: item[0])[1]
                        counts["nearest_ahead_correct" if nearest == upcoming else "nearest_ahead_wrong"] += 1
                    else:
                        counts["no_opening_ahead"] += 1

                if front - 0.30 < position[1] <= doors[upcoming][2] + 0.28:
                    counts["near_plane_scans"] += 1
                    counts["tracked_near_plane_correct"] += tracked_door == upcoming
                    counts["near_plane_next_door_found"] += any(door == upcoming for _, door in matches)

            for index, (_, _, back, _) in enumerate(doors):
                if position[1] < back + 0.28:
                    continue
                event = passes[index]
                if event["ground_truth_time_s"] is None:
                    event["ground_truth_time_s"] = t
                if event["first_rear_observation_s"] is None and any(door == index for _, door in matches):
                    event["first_rear_observation_s"] = t

    for event in reported:
        index = event["matched_door"]
        real = passes[index]["ground_truth_time_s"] if index is not None else None
        event["delta_from_pass_s"] = event["time_s"] - real if real is not None else None
    return counts, passes, reported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("batches", type=Path, nargs="+")
    args = parser.parse_args()
    total = Counter()
    cases = {}
    for batch in args.batches:
        walls_by_case = {(int(seed), int(task)): rects(walls) for seed, task, _, _, walls, _, _
                         in LAYOUT.findall((batch / "unity.log").read_text())}
        for path in sorted((batch / "recordings").glob("*.csv")):
            outcome, family, task, seed = CASE.match(path.name).groups()
            walls = walls_by_case[int(seed), int(task)]
            doors = []
            for left, right in zip(walls[4::2], walls[5::2]):
                edge = left[0] + left[2]
                doors.append(((edge + right[0]) / 2, left[1], left[1] + left[3], right[0] - edge))
            if len(doors) != 2:
                raise ValueError(f"Expected two annotated doors for {path}")
            counts, passes, reported = score(path, doors)
            cases[f"{seed}/{task}"] = {"outcome": outcome, "family": family,
                                     "counts": dict(counts), "passes": passes, "reported": reported}
            total.update(counts)

    events = [event for case in cases.values() for event in case["passes"]
              if event["ground_truth_time_s"] is not None]
    delays = [event["first_rear_observation_s"] - event["ground_truth_time_s"] for event in events
              if event["first_rear_observation_s"] is not None]
    tracked_events = [event for case in cases.values() for event in case["reported"]]
    detections = sum(any(r["matched_door"] == index and r["delta_from_pass_s"] is not None
                         and abs(r["delta_from_pass_s"]) <= 1.0 + 1e-6 for r in case["reported"])
                     for case in cases.values() for index in range(len(case["passes"])))
    false_alarms = sum(r["delta_from_pass_s"] is None or abs(r["delta_from_pass_s"]) > 1.0 + 1e-6
                       for r in tracked_events)
    totals = {"episodes": len(cases), **total, "passing_events": len(events),
              "matched_door_reobserved_within_1s": sum(delay <= 1.0 + 1e-6 for delay in delays),
              "matched_door_reobserved_at_any_later_time": len(delays),
              "tracker_reported_events": len(tracked_events),
              "tracker_matched_pass_within_1s": detections, "tracker_false_or_late_events": false_alarms}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"totals": totals, "cases": cases}, indent=2) + "\n")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
