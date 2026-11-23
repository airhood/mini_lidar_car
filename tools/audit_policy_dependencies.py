"""Test how a saved lower policy responds to input interventions on held-out Unity recordings."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import re

import torch

from train_lower_policy import HISTORY, ScanDataset

SEED = re.compile(r"-seed(\d+)-")
LABEL = re.compile(r"-(follow|left|right)-task\d+")


def held_out(manifest, recordings, per_group):
    groups = defaultdict(list)
    for name in manifest.read_text().splitlines():
        seed, label = SEED.search(name), LABEL.search(name)
        if not seed or not label or int(seed[1]) % 5 != 4:
            continue
        group = "blocked" if name.startswith("stopped-") else label[1]
        groups[group].append(recordings / name)
    randomizer = random.Random(0)
    return [path for group, paths in sorted(groups.items())
            for path in randomizer.sample(paths, min(per_group, len(paths)))]


@torch.inference_mode()
def probe(model, dataset):
    row_indices = torch.arange(len(dataset))
    history = (row_indices[:, None] - torch.arange(HISTORY - 1, -1, -1)).maximum(dataset.starts[:, None])
    windows = dataset.scans[history]
    intents = dataset.intents
    previous = dataset.previous
    target = dataset.commands

    def evaluate(windows, prior):
        return torch.cat([model(scans, intent, commands) for scans, intent, commands in
                          zip(windows.split(128), intents.split(128), prior.split(128))])

    normal = evaluate(windows, previous)
    generator = torch.Generator().manual_seed(0)
    shuffled = row_indices.clone()
    for intent in range(4):
        group = row_indices[intents == intent]
        shuffled[group] = group[torch.randperm(len(group), generator=generator)]
    swapped_scan = evaluate(windows[shuffled], previous)
    zero_previous = evaluate(windows, torch.zeros_like(previous))
    shifted_turn = evaluate(windows, (previous + torch.tensor([0.04, -0.04])).clamp(-1, 1))

    replay = torch.empty_like(target)
    starts = dataset.starts.unique()
    lengths = torch.diff(torch.cat((starts, torch.tensor([len(dataset)]))))
    previous_output = torch.zeros(len(starts), 2)
    for step in range(int(lengths.max())):
        active = lengths > step
        at = starts[active] + step
        result = model(windows[at], intents[at], previous_output[active])
        replay[at] = result
        previous_output[active] = result

    turn_delta = ((target[:, 0] - target[:, 1]) - (previous[:, 0] - previous[:, 1])) * 0.5
    teacher_transition = turn_delta.abs() >= 0.04
    teacher_transition[starts] = False
    cases = {
        "repeat_previous_teacher_command": previous,
        "normal": normal,
        "same_intent_shuffled_scans": swapped_scan,
        "previous_command_set_to_zero": zero_previous,
        "previous_turn_shifted_by_0_04": shifted_turn,
        "recursive_command_replay_with_recorded_scans": replay,
    }

    def mean_abs(a, b):
        return round(float((a - b).abs().mean()), 6)

    return {
        "rows": len(dataset),
        "active_turn_transition_rows": int(teacher_transition.sum()),
        "cases": {
            name: {
                "mae_to_teacher": mean_abs(pred, target),
                "mae_to_teacher_during_turn_transition": mean_abs(pred[teacher_transition], target[teacher_transition])
                if teacher_transition.any() else None,
                "mean_abs_change_from_normal_output": mean_abs(pred, normal),
            } for name, pred in cases.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("recordings", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--per-group", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.per_group < 1:
        parser.error("--per-group must be positive")

    torch.set_num_threads(min(4, torch.get_num_threads()))
    paths = held_out(args.manifest, args.recordings, args.per_group)
    if not paths:
        parser.error("no recordings from the v8 validation seeds")
    model = torch.jit.load(str(args.checkpoint), map_location="cpu").eval()
    result = probe(model, ScanDataset(paths))
    result["trainable_parameters"] = sum(tensor.numel() for tensor in model.parameters() if tensor.requires_grad)
    result["recordings"] = [path.name for path in paths]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "recordings"}, indent=2))


if __name__ == "__main__":
    main()
