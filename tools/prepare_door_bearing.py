"""Replace fixed direction labels with bearings derived only from static doors."""

import argparse
import csv
import json
import math
from pathlib import Path

from inspect_direction_transfer import CASE, read_batch


def door_rows(walls):
    if len(walls) != 8:
        raise ValueError("Expected perimeter and two solid wall pairs")
    result = []
    for left, right in zip(walls[4::2], walls[5::2]):
        left_x, left_z, left_width, left_length = left
        right_x, right_z, _, right_length = right
        if left_z != right_z or left_length != right_length or left_x + left_width >= right_x:
            raise ValueError("Invalid static door pair")
        result.append(((left_x + left_width + right_x) / 2, left_z + left_length))
    return result


def target(x, z, doors, exit_z=None):
    for center, back in doors:
        if z <= back + 0.28:
            dx, dz = center - x, back + 0.60 - z
            length = math.hypot(dx, dz)
            return x + dx / length, z + dz / length
    if exit_z is not None:
        dx, dz = -x, exit_z + 0.60 - z
        length = math.hypot(dx, dz)
        if length > 0.00001:
            return x + dx / length, z + dz / length
    return x, z + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--continue-to-exit", action="store_true")
    args = parser.parse_args()
    layouts, cases = read_batch(args.teacher)
    args.output.mkdir(parents=True, exist_ok=True)
    audited = {}
    for path in sorted((args.teacher / "recordings").glob("completed-*.csv")):
        match = CASE.match(path.name)
        key = int(match[4]), int(match[3])
        family, _, walls, _, (exit_z, _) = layouts[key]
        if cases[key]["outcome"] != "completed":
            raise ValueError(f"Expected completed teacher recording: {path}")
        doors = door_rows(walls)
        count = 0
        with path.open(newline="") as source, (args.output / path.name).open("x", newline="") as dest:
            reader = csv.DictReader(source)
            writer = csv.DictWriter(dest, reader.fieldnames, lineterminator="\n")
            writer.writeheader()
            for row in reader:
                x, z = float(row["x_m"]), float(row["z_m"])
                wx, wz = target(x, z, doors, exit_z if args.continue_to_exit else None)
                row["waypoint_x_m"] = f"{wx:.10f}"
                row["waypoint_z_m"] = f"{wz:.10f}"
                writer.writerow(row)
                count += 1
        audited[f"{key[0]}/{key[1]}"] = {"family": family, "scans": count, "doors": doors, "exit_z": exit_z if args.continue_to_exit else None}
    if not audited:
        raise ValueError("No completed door runs")
    (args.output.parent / "bearing-labels.json").write_text(json.dumps(audited, indent=2) + "\n")
    print(f"Added door bearings to {len(audited)} existing teacher runs")


if __name__ == "__main__":
    main()
