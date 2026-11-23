"""Plot held-out trajectories and measure the response to swapping only the first scan."""

import argparse
import json
from pathlib import Path

import numpy as np

from summarize_side_swap import load_batch

LABELS = {"teacher-heldout": "Teacher", "lidar": "LiDAR", "blind": "LiDAR masked"}


def initial_steering(root, batches):
    import torch

    torch.set_num_threads(2)
    teacher = batches["teacher-heldout"]
    keys = sorted(teacher)
    beam_names = [f"range_{beam}_m" for beam in range(360)]
    ranges = torch.tensor([[float(teacher[key]["samples"][0][name]) for name in beam_names]
                           for key in keys]).clamp(0, 8) / 8
    histories = ranges[:, None, :].expand(-1, 4, -1)
    direction = torch.tensor([[1.0, 0.0]]).expand(len(keys), -1)
    intents = torch.zeros(len(keys), dtype=torch.long)
    results = {}
    for arm in ("lidar", "blind"):
        policy = torch.jit.load(str(root / arm / "lower-policy.pt")).eval()
        with torch.inference_mode():
            commands = policy(histories if arm == "lidar" else torch.ones_like(histories),
                              intents, direction)
        steering = (commands[:, 0] - commands[:, 1]).tolist()
        paired = {f"{seed}": {"box_left": steering[i], "box_right": steering[i + 1]}
                  for i, (seed, _) in enumerate(keys) if i % 2 == 0}
        results[arm] = {
            "rightward_steering_increases_for_box_left": sum(
                pair["box_left"] - pair["box_right"] > 1e-6 for pair in paired.values()),
            "steering_changes_in_correct_direction_on_both_sides": sum(
                pair["box_left"] > 0 > pair["box_right"] for pair in paired.values()),
            "pairs": paired,
        }
    (root / "initial-steering.json").write_text(json.dumps(results, indent=2) + "\n")
    print({arm: {key: value for key, value in result.items() if key != "pairs"}
           for arm, result in results.items()})


def plot(root, batches, layouts):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    seeds = sorted({seed for seed, _ in layouts})
    fig, axes = plt.subplots(2, 2, figsize=(9, 9), layout="constrained")
    for row, seed in enumerate((seeds[0], seeds[-1])):
        for column, side in enumerate(("left", "right")):
            ax = axes[row, column]
            area, box, exit_z = layouts[seed, side]
            x, z, width, length = area
            ax.add_patch(Rectangle((x, z), width, length, fill=False, edgecolor="0.4"))
            ax.add_patch(Rectangle(box[:2], box[2], box[3], facecolor="0.7"))
            ax.plot([-0.5, 0.5], [exit_z, exit_z], linestyle=":", color="0.4", label="Exit gate")
            for arm, label in LABELS.items():
                case = batches[arm][seed, side]
                xy = np.array([(float(sample["x_m"]), float(sample["z_m"]))
                               for sample in case["samples"]])
                line, = ax.plot(*xy.T, label=label, linestyle="--" if arm == "teacher-heldout" else "-")
                ax.scatter(*xy[-1], marker="x", color=line.get_color(), s=30)
            ax.set(aspect="equal", xlim=(x - 0.12, x + width + 0.12),
                   ylim=(z - 0.12, z + length + 0.12), xlabel="x (m)", ylabel="z (m)",
                   title=f"Seed {seed} | box on {side}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=4, fontsize=8)
    fig.savefig(root / "measured-trajectories.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    batches = {arm: load_batch(args.directory / arm)[0] for arm in LABELS}
    _, layouts = load_batch(args.directory / "teacher-heldout")
    initial_steering(args.directory, batches)
    plot(args.directory, batches, layouts)


if __name__ == "__main__":
    main()
