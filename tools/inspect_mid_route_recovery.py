"""Audit paired teacher restarts and score sequential obstacle evaluations."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re

from inspect_direction_transfer import read_batch
from prepare_mid_route_recovery import heading_difference
from train_lower_policy import paired_recovery_paths

RESTART = re.compile(r"^(completed|timeout|collision)-teacher-v12-recovery-(normal|policy)-(?:staggered|angled-entry|wall-side)-follow-task(\d+)-seed(\d+)-")


def audit_restarts(root):
    with (root / "recovery-starts.csv").open(newline="") as file:
        starts = {(int(row["seed"]), int(row["task"]), row["source"]): row
                  for row in csv.DictReader(file)}
    cases = {}
    families = {}
    for path in sorted((root / "restart-teacher" / "recordings").glob("*.csv")):
        match = RESTART.match(path.name)
        if match is None:
            raise ValueError(f"Unexpected restart file: {path}")
        outcome, source, task, seed = match.groups()
        key = (int(seed), int(task), source)
        if key in cases:
            raise ValueError(f"Repeated restart: {key}")
        with path.open(newline="") as stream:
            first = next(csv.DictReader(stream))
        start = starts[key]
        if (any(abs(float(first[field]) - float(start[field])) > 0.001 for field in ("x_m", "z_m"))
                or heading_difference(float(first["heading_deg"]), float(start["heading_deg"])) > 0.1):
            raise ValueError(f"Unexpected first pose for {key}")
        cases[key] = outcome
        families[key] = path.name.split(f"-recovery-{source}-", 1)[1].split("-follow-", 1)[0]
    if starts.keys() != cases.keys():
        raise ValueError("Restart recordings do not match the manifest")
    counts = {source: dict(Counter(outcome for (*_, kind), outcome in cases.items() if source == kind))
              for source in ("normal", "policy")}
    directory = root / "restart-teacher" / "recordings"
    paired = {kind: paired_recovery_paths(directory, kind) for kind in counts}
    if not paired["normal"] or len(paired["normal"]) != len(paired["policy"]):
        raise ValueError("Expected equal, nonempty matched training pairs")
    pairs_by_family = Counter(families[seed, task, kind] for seed, task, kind in cases
                              if kind == "normal" and cases[(seed, task, "normal")] == "completed"
                              and cases.get((seed, task, "policy")) == "completed")
    (root / "restart-audit.json").write_text(json.dumps({
        "teacher_outcomes": counts,
        "added_training_runs_per_model": len(paired["normal"]),
        "complete_pairs_by_family": dict(pairs_by_family),
    }, indent=2) + "\n")
    print("Restart audit", counts, f"({len(paired['normal'])} training pairs)")


def stage_metrics(cases):
    return {
        "outcomes": dict(Counter(case["outcome"] for case in cases.values())),
        "first_box_cleared": sum(case["boxes_cleared"] >= 1 for case in cases.values()),
        "second_box_cleared": sum(case["boxes_cleared"] >= 2 for case in cases.values()),
        "third_box_cleared": sum(case["boxes_cleared"] >= 3 for case in cases.values()),
        "collisions_after_first_box": sum(case["outcome"] == "collision" and case["boxes_cleared"] >= 1
                                          for case in cases.values()),
    }


def evaluate(root, transfer):
    results = {}
    traces = {}
    for cohort in ("previous", "new"):
        expected, teacher = read_batch(transfer / "teacher-heldout" if cohort == "previous"
                                       else root / "teacher-new")
        expected = {key: layout for key, layout in expected.items() if layout[0] == "staggered"}
        if len(expected) != 10 or any(teacher[key]["outcome"] != "completed" for key in expected):
            raise ValueError(f"Expected ten feasible sequential cases in {cohort}")
        traces[cohort] = (expected, {"teacher": {key: teacher[key] for key in expected}})
        results[cohort] = {}
        for arm in ("baseline", "normal", "policy"):
            path = transfer / "mixed-lidar" if cohort == "previous" and arm == "baseline" else (
                root / f"{arm}-{cohort}")
            layouts, cases = read_batch(path)
            layouts = {key: layout for key, layout in layouts.items() if layout[0] == "staggered"}
            if layouts != expected:
                raise ValueError(f"Changed geometry in {cohort}/{arm}")
            cases = {key: cases[key] for key in expected}
            for key in expected:
                if cases[key]["positions"][0] != teacher[key]["positions"][0]:
                    raise ValueError(f"Changed start position in {cohort}/{arm}/{key}")
            traces[cohort][1][arm] = cases
            results[cohort][arm] = {
                "summary": stage_metrics(cases),
                "cases": {f"{seed}/{task}": {k: v for k, v in case.items() if k != "positions"}
                          for (seed, task), case in sorted(cases.items())},
            }
            print(cohort, arm, results[cohort][arm]["summary"])
    (root / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    plot(root, traces)


def plot(root, traces):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    names = {"teacher": "Teacher", "baseline": "Previous model",
             "normal": "Normal restarts", "policy": "Policy-pose restarts"}
    fig, axes = plt.subplots(1, 2, figsize=(8, 7), layout="constrained")
    for ax, (cohort, (layouts, policies)) in zip(axes, traces.items()):
        key = min(layouts, key=lambda key: (
            policies["policy"][key]["outcome"] != "completed",
            policies["normal"][key]["outcome"] == "completed", key))
        _, area, walls, boxes, (goal, width) = layouts[key]
        for rect in walls:
            ax.add_patch(Rectangle(rect[:2], *rect[2:], color="0.25", lw=0))
        for rect in boxes:
            ax.add_patch(Rectangle(rect[:2], *rect[2:], color="0.66", lw=0))
        ax.plot([-width, width], [goal, goal], ":", color="0.5", label="Exit")
        for arm, cases in policies.items():
            x, z = zip(*cases[key]["positions"])
            line, = ax.plot(x, z, label=names[arm], linestyle="--" if arm == "teacher" else "-")
            ax.scatter(x[-1], z[-1], color=line.get_color(), marker="x")
        x, z, width, length = area
        ax.set(xlim=(x - .17, x + width + .17), ylim=(z - .17, z + length + .17),
               xlabel="x (m)", ylabel="z (m)", aspect="equal",
               title=f"{cohort.title()} evaluation | seed {key[0]}, task {key[1]}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, fontsize=8)
    fig.savefig(root / "measured-trajectories.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--transfer", type=Path)
    args = parser.parse_args()
    audit_restarts(args.directory)
    if args.transfer:
        evaluate(args.directory, args.transfer)
