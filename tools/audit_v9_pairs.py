"""Check matched clear and obstacle runs from a single Unity collection."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

import numpy as np

from audit_unity_recordings import read_episode, summary

EPISODE = re.compile(
    r"(?P<status>completed|stopped|collision|timeout|interrupted)-teacher-v(?P<version>9|10|11|12)-"
    r"(?P<family>network|rooms|hall-[a-z]+)-(?P<intent>follow|left|right)-"
    r"(?P<placement>clear|box-approach|box-exit|blocked|clutter\d+)-task(?P<task>\d+)-seed(?P<seed>\d+)-"
    r"\d{8}-\d{6}-[\da-f]+\.csv"
)


def audit(directory, first_seed=None, seed_count=None, version=9):
    outcomes = Counter()
    placements = Counter()
    placements_by_family = defaultdict(Counter)
    usable_by_seed = Counter()
    pairs = defaultdict(dict)
    failed = []
    interval_errors = 0
    scan_count = 0
    contact_runs = 0

    for path in sorted(directory.glob("*.csv")):
        match = EPISODE.fullmatch(path.name)
        if not match:
            continue
        run = match.groupdict()
        if int(run["version"]) != version:
            continue
        seed = int(run["seed"])
        if first_seed is not None and not first_seed <= seed < first_seed + seed_count:
            continue
        outcomes[run["status"]] += 1
        placements[run["placement"]] += 1
        try:
            rows = read_episode(path)
        except (ValueError, OSError) as error:
            failed.append({"file": path.name, "error": str(error)})
            continue
        scan_count += len(rows)
        gap_count = int(np.count_nonzero(np.abs(np.diff(rows[:, 0]) - 0.1) > 0.002))
        interval_errors += gap_count
        collision = bool(np.any(rows[:, 6] > 0))
        contact_runs += int(collision)

        if not collision and not gap_count and run["status"] in ("completed", "stopped"):
            usable_by_seed[seed] += 1
            placements_by_family[run["family"]][run["placement"]] += 1

        if run["status"] == "completed" and not collision and not gap_count:
            key = (seed, int(run["task"]), run["intent"], run["family"])
            role = "clear" if run["placement"] == "clear" else "box"
            if role in pairs[key]:
                raise ValueError(f"Duplicate {role} for paired mission {key}")
            pairs[key][role] = rows[:, 3:5].copy()

    matched = []
    for (seed, task, intent, family), pair in sorted(pairs.items()):
        if pair.keys() != {"clear", "box"}:
            continue
        clear, box = pair["clear"], pair["box"]
        differences = box[:, None, :] - clear[None, :, :]
        nearest = np.sqrt(np.square(differences).sum(axis=2).min(axis=1))
        matched.append({
            "seed": seed, "task": task, "intent": intent, "family": family,
            "start_difference_m": float(np.linalg.norm(box[0] - clear[0])),
            "max_deviation_from_clear_m": float(np.max(nearest)),
            "finish_difference_m": float(np.linalg.norm(box[-1] - clear[-1])),
        })
    return {
        "outcomes": dict(outcomes), "placements": dict(placements), "scans": scan_count,
        "usable_placements_by_family": {family: dict(counts) for family, counts in placements_by_family.items()},
        "usable_seed_count": len(usable_by_seed),
        "irregular_intervals": interval_errors, "contact_runs": contact_runs, "invalid_files": failed,
        "complete_pairs": len(matched), "unmatched_complete_runs": len(pairs) - len(matched),
        "max_deviation_from_clear_m": summary([m["max_deviation_from_clear_m"] for m in matched]),
        "pairs": matched,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-seed", type=int)
    parser.add_argument("--seed-count", type=int)
    parser.add_argument("--version", type=int, choices=(9, 10, 11, 12), default=9)
    args = parser.parse_args()
    if (args.first_seed is None) != (args.seed_count is None):
        parser.error("--first-seed and --seed-count must be supplied together")
    if args.first_seed is not None and (args.first_seed < 0 or args.seed_count < 1):
        parser.error("require --first-seed >= 0 and --seed-count >= 1")
    report = audit(args.recordings, args.first_seed, args.seed_count, args.version)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
