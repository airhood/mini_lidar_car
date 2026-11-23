"""Compare route types and recorded LiDAR and command distributions across Unity batches."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

from audit_unity_recordings import read_episode

NAME = re.compile(
    r"(?P<outcome>completed|stopped|collision|timeout|interrupted)-teacher-v(?P<version>\d+)-"
    r"(?P<family>network|rooms|hall-[a-z]+)-(?P<intent>follow|left|right)"
    r"(?:-(?P<placement>clear|box-approach|box-exit|blocked|clutter\d+))?"
    r"-task(?P<task>\d+)-seed(?P<seed>\d+)-\d{8}-\d{6}-[\da-f]+\.csv"
)
FRONT = np.r_[345:360, :16]
INTENTS = ("Follow", "Left", "Right", "Stop")
ENVIRONMENTS = ("Corridor", "Narrow room", "Hall / linked room")
METRICS = {
    "front": ("Front range p10 (m)", np.linspace(0, 6, 31)),
    "side_delta": ("Absolute left/right range difference (m)", np.linspace(0, 6, 31)),
    "missing": ("Missing LiDAR beams (%)", np.linspace(0, 100, 41)),
    "turn": ("Signed motor turn command (left - right) / 2", np.linspace(-0.24, 0.24, 49)),
}


def collect(paths, version):
    episodes = []
    outcomes = Counter()
    invalid = []
    for path in paths:
        match = NAME.fullmatch(path.name)
        if match is None or int(match["version"]) != version:
            continue
        info = match.groupdict()
        outcomes[info["outcome"]] += 1
        if info["outcome"] not in ("completed", "stopped"):
            continue
        try:
            rows = read_episode(path)
            intervals = np.diff(rows[:, 0])
            scans = rows[:, 8:]
            if (np.any(rows[:, 6] != 0) or np.any(np.abs(intervals - 0.1) > 0.002)
                    or np.any(~np.isin(rows[:, 7], (0, 1, 2, 3))) or not np.all(scans >= 0)):
                raise ValueError("invalid timing, contact, intent or scan")
        except (OSError, ValueError) as error:
            invalid.append(f"{path.name}: {error}")
            continue

        sampled = scans[::5]
        ranges = np.minimum(sampled, 6.0)
        metrics = {
            "front": np.percentile(ranges[:, FRONT], 10, axis=1),
            "side_delta": np.abs(np.median(ranges[:, 45:106], axis=1) -
                                 np.median(ranges[:, 255:316], axis=1)),
            "missing": np.isinf(sampled).mean(axis=1) * 100,
            "turn": (rows[:, 1] - rows[:, 2]) / 2,
        }
        for key in ("seed", "task"):
            info[key] = int(info[key])
        info.update(metrics=metrics, frames=len(rows),
                    intents=np.bincount(rows[:, 7].astype(int), minlength=4))
        episodes.append(info)
    if not episodes:
        raise ValueError(f"No usable v{version} episodes")
    return episodes, dict(outcomes), invalid


def environment(episode):
    return {"network": "Corridor", "rooms": "Narrow room"}.get(episode["family"], "Hall / linked room")


def aggregate(episodes):
    if not episodes:
        return {"episodes": 0}
    intent_counts = np.sum([episode["intents"] for episode in episodes], axis=0)
    frames = int(intent_counts.sum())
    intent_pct = np.round(intent_counts / frames * 100, 2)
    metrics = {}
    for key in METRICS:
        values = np.concatenate([episode["metrics"][key] for episode in episodes])
        q10, median, q90 = np.percentile(values, (10, 50, 90))
        metrics[key] = {"p10": round(float(q10), 4), "median": round(float(median), 4), "p90": round(float(q90), 4)}
        if key == "turn":
            metrics[key]["abs_ge_0_08_pct"] = round(float(np.mean(np.abs(values) >= 0.08) * 100), 2)
    return {
        "episodes": len(episodes),
        "frames": frames,
        "seeds": sorted({episode["seed"] for episode in episodes}),
        "environments": dict(Counter(map(environment, episodes))),
        "families": dict(Counter(episode["family"] for episode in episodes)),
        "recording_outcomes": dict(Counter(episode["outcome"] for episode in episodes)),
        "placements": dict(Counter(episode["placement"] or "v8 protocol" for episode in episodes)),
        "intent_frame_pct": dict(zip(INTENTS, intent_pct.tolist())),
        "measurement_distributions": metrics,
    }


def distribution_plots(groups, title, output):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
    for ax, (key, (label, bins)) in zip(axes.flat, METRICS.items()):
        for name, episodes in groups.items():
            values = np.clip(np.concatenate([episode["metrics"][key] for episode in episodes]), bins[0], bins[-1])
            ax.hist(values, bins=bins, weights=np.full(values.size, 100 / values.size),
                    histtype="step", label=f"{name} ({len(episodes)} runs)")
        ax.set(xlabel=label, ylabel="Share per bin (%)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def coverage_plot(groups, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    positions = np.arange(len(groups))
    for ax, key, labels, title in (
        (axes[0], "environments", ENVIRONMENTS, "Saved episodes by environment"),
        (axes[1], "intent_frame_pct", INTENTS, "Recorded instruction labels"),
    ):
        bottom = np.zeros(len(groups))
        for label in labels:
            values = np.array([entry[key].get(label, 0) for entry in groups.values()], dtype=float)
            if key == "environments":
                values = values / np.array([entry["episodes"] for entry in groups.values()]) * 100
            ax.bar(positions, values, bottom=bottom, label=label)
            bottom += values
        ax.set(xticks=positions, xticklabels=list(groups), ylim=(0, 100), ylabel="Share (%)", title=title)
        ax.legend(fontsize=8)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v8", type=Path, required=True, help="directory containing v8 CSVs")
    parser.add_argument("--v8-manifest", type=Path, required=True, help="v8 training filenames")
    for version in (9, 10, 12):
        parser.add_argument(f"--v{version}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    groups, results = {}, {}
    for version in (8, 9, 10, 12):
        name = f"v{version}"
        folder = getattr(args, name)
        paths = (folder / line for line in args.v8_manifest.read_text().splitlines() if line) if version == 8 else sorted(folder.glob("*.csv"))
        episodes, outcomes, invalid = collect(paths, version)
        groups[name] = episodes
        results[name] = aggregate(episodes) | {"listed_outcomes": outcomes, "invalid_or_contact_files": invalid}

    common = {episode["seed"] for episode in groups["v10"]} & {episode["seed"] for episode in groups["v12"]}
    comparable = {name: [ep for ep in groups[name] if ep["seed"] in common and ep["family"] == "network"]
                  for name in ("v10", "v12")}
    latest = {label: [ep for ep in groups["v12"] if pred(ep)] for label, pred in (
        ("Corridor with box", lambda ep: ep["family"] == "network" and ep["placement"].startswith("box-")),
        ("Hall with obstacles", lambda ep: ep["family"].startswith("hall-") and ep["placement"].startswith("clutter")),
    )}
    report = {
        "batches": results,
        "shared_seed_corridors": {label: aggregate(episodes) for label, episodes in comparable.items()},
        "v12_obstacle_conditions": {label: aggregate(episodes) for label, episodes in latest.items()},
        "sampling": "LiDAR uses every fifth scan within each episode; controls and intent counts use all scans.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    coverage_plot(results, args.output / "coverage.png")
    distribution_plots(groups, "Recorded distributions across batches", args.output / "distributions.png")
    distribution_plots(comparable, "Corridor recordings on shared seeds", args.output / "shared-seed-corridors.png")
    distribution_plots(latest, "v12 obstacle runs by environment", args.output / "v12-obstacle-conditions.png")
    print(json.dumps({name: {key: result[key] for key in ("episodes", "frames", "environments", "intent_frame_pct")}
                      for name, result in results.items()}, indent=2))


if __name__ == "__main__":
    main()
