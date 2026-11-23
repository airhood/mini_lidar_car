"""Score the recovery candidate on original courses and independently seeded harder courses."""

import argparse
from collections import Counter
import json
from pathlib import Path

from inspect_direction_transfer import read_batch


def compare(overview, teacher_path, previous_path, candidate_path, expected):
    teacher_layouts, teacher = read_batch(teacher_path)
    layouts, previous = read_batch(previous_path)
    candidate_layouts, candidate = read_batch(candidate_path)
    if len(layouts) != expected or candidate_layouts != layouts or teacher_layouts.keys() - layouts.keys():
        raise ValueError(f"Mismatch in requested courses for {overview}")
    eligible = {key for key, case in teacher.items() if case["outcome"] == "completed"}
    if not eligible:
        raise ValueError(f"No teacher completed courses for {overview}")
    for key in teacher:
        if teacher_layouts[key] != layouts[key]:
            raise ValueError(f"Changed course for {overview}/{key}")
        for case in (previous[key], candidate[key]):
            if (case["positions"][0] != teacher[key]["positions"][0]
                    or case["start_heading"] != teacher[key]["start_heading"]):
                raise ValueError(f"Changed start pose for {overview}/{key}")
    summary = {}
    cases = {}
    for family in sorted({layout[0] for layout in layouts.values()}):
        keys = [key for key in sorted(layouts) if layouts[key][0] == family]
        passed_teacher = [key for key in keys if key in eligible]
        summary[family] = {
            "requested": len(keys),
            "teacher_completed": len(passed_teacher),
            "previous": dict(Counter(previous[key]["outcome"] for key in passed_teacher)),
            "candidate": dict(Counter(candidate[key]["outcome"] for key in passed_teacher)),
        }
        for seed, task in keys:
            key = seed, task
            cases[f"{seed}/{task}"] = {
                "family": family,
                "teacher": teacher[key]["outcome"] if key in teacher else "skipped",
                **{name: {field: value for field, value in arm[key].items() if field != "positions"}
                   for name, arm in (("previous", previous), ("candidate", candidate))},
            }
    totals = {
        "requested": len(layouts),
        "teacher_completed": len(eligible),
        "previous": dict(Counter(previous[key]["outcome"] for key in eligible)),
        "candidate": dict(Counter(candidate[key]["outcome"] for key in eligible)),
        "candidate_only_completed": sum(candidate[key]["outcome"] == "completed"
                                        and previous[key]["outcome"] != "completed" for key in eligible),
        "previous_only_completed": sum(previous[key]["outcome"] == "completed"
                                       and candidate[key]["outcome"] != "completed" for key in eligible),
    }
    print(overview, totals)
    for family, counts in summary.items():
        print(f"  {family}: {counts}")
    return {"totals": totals, "by_family": summary, "cases": cases}, (
        layouts, teacher, previous, candidate, eligible)


def plot(output, data):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    layouts, teacher, previous, candidate, eligible = data
    families = sorted({layout[0] for layout in layouts.values()})
    fig, axes = plt.subplots(1, len(families), figsize=(11, 6), layout="constrained")
    for ax, family in zip(axes, families):
        keys = [key for key in eligible if layouts[key][0] == family]
        if not keys:
            ax.set_axis_off()
            continue
        key = min(keys, key=lambda key: (
            (candidate[key]["outcome"] == "completed") == (previous[key]["outcome"] == "completed"),
            candidate[key]["outcome"] == "completed",
            key,
        ))
        _, area, walls, boxes, (exit_z, half_width) = layouts[key]
        for rect in walls:
            ax.add_patch(Rectangle(rect[:2], *rect[2:], color="0.30", lw=0))
        for rect in boxes:
            ax.add_patch(Rectangle(rect[:2], *rect[2:], color="0.65", lw=0))
        ax.plot([-half_width, half_width], [exit_z, exit_z], ":", color="0.4", label="Exit")
        for label, rows in (("Teacher", teacher), ("Previous model", previous), ("Candidate", candidate)):
            x, z = zip(*rows[key]["positions"])
            line, = ax.plot(x, z, "--" if label == "Teacher" else "-", label=label)
            ax.scatter(x[-1], z[-1], s=26, marker="x", color=line.get_color())
        x, z, width, height = area
        ax.set(aspect="equal", xlim=(x - .17, x + width + .17), ylim=(z - .17, z + height + .17),
               title=f"{family} | {key[0]}/{key[1]}", xlabel="x (m)", ylabel="z (m)")
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="outside lower center", ncols=4)
            break
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("original", type=Path)
    args = parser.parse_args()
    run, reference = args.experiment, args.original
    original, _ = compare("Original", reference / "teacher-heldout", reference / "recovery-lidar",
                          run / "candidate-original", 35)
    new, traces = compare("Independent stress seeds", run / "teacher-new", run / "previous-new",
                          run / "candidate-new", 24)
    (run / "summary.json").write_text(json.dumps({"original": original, "new": new}, indent=2) + "\n")
    plot(run / "new-course-trajectories.png", traces)


if __name__ == "__main__":
    main()
