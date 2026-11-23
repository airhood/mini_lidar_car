"""Extract decision approaches from paired, successfully completed Unity teacher runs."""

import argparse
import csv
import json
from pathlib import Path

from evaluate_direction_choice import CASE, SIDE, read_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    layouts, runs = read_batch(args.teacher)
    recordings = {}
    for path in sorted((args.teacher / "recordings").glob("*.csv")):
        _, family, side, seed = CASE.match(path.name).groups()
        recordings[int(seed), family, side] = path

    kept = {}
    omitted = []
    for seed, family in sorted({key[:2] for key in runs}):
        left, right = (seed, family, "left"), (seed, family, "right")
        if layouts[left] != layouts[right] or runs[left]["start"] != runs[right]["start"]:
            raise ValueError(f"Paired geometry, pose or LiDAR mismatch for {seed}/{family}")
        if not all(runs[key]["outcome"] == "completed" and runs[key]["actual"] == key[-1]
                   for key in (left, right)):
            omitted.append(f"{seed}/{family}")
            continue
        for key in (left, right):
            start_z = runs[key]["positions"][0][1]
            front_z = layouts[key][1][1]
            if not 0.65 <= front_z - start_z <= 1.01:
                raise ValueError(f"Expected close approach start for {key}")
            path = recordings[key]
            with path.open(newline="") as file:
                reader = csv.reader(file)
                header = next(reader)
                z_column = header.index("z_m")
                rows = []
                for row in reader:
                    if float(row[z_column]) >= front_z:
                        break
                    rows.append(row)
            if len(rows) < 4:
                raise ValueError(f"Too few approach scans for {key}")
            kept[key] = (path.name, header, rows)

    if not kept:
        raise ValueError("No eligible paired approaches")
    recordings_dir = args.target / "recordings"
    recordings_dir.mkdir(parents=True)
    audit = {}
    for (seed, family, side), (name, header, rows) in kept.items():
        with (recordings_dir / name).open("x", newline="") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
        audit[f"{seed}/{family}/{side}"] = {"file": name, "windows": len(rows),
                                           "split": "validation" if seed % 5 == 4 else "train"}
    result = {
        "eligible_pairs": len(kept) // len(SIDE), "omitted_pairs": omitted, "episodes": audit,
        "windows": {split: sum(row["windows"] for row in audit.values() if row["split"] == split)
                    for split in ("train", "validation")},
    }
    (args.target / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
