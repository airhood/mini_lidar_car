"""Train the LiDAR-based lower policy from Unity teacher CSV recordings."""

import argparse
import csv
import math
from pathlib import Path
import re

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

BEAMS = 360
HISTORY = 4
INTENTS = 4
RANGE_LIMIT_M = 8.0
PERIOD_S = 0.1
GOAL_SCALE_M = 15.0
HALL_CASE = re.compile(r"teacher-v12-(hall-[a-z]+)-follow-(?:clear|clutter\d+)-task(\d+)-seed(\d+)-")
FILE_NAME = re.compile(r"^(?:completed|stopped)-teacher-v(?P<version>\d+)-(?P<layout>.+?)-\d{8}-\d{6}-[\da-f]+\.csv$")
DEFAULT_HOLDOUTS = ()
RANGE_COLUMNS = tuple(f"range_{beam}_m" for beam in range(BEAMS))


class LowerPolicy(nn.Module):
    def __init__(self, use_previous_command=True, use_relative_goal=False, temporal=False, history_stride=1):
        super().__init__()
        self.use_previous_command = use_previous_command
        self.use_relative_goal = use_relative_goal
        self.num_intents = INTENTS
        self.temporal = temporal
        self.history_stride = history_stride
        self.scan_encoder = nn.Sequential(
            nn.Conv1d(HISTORY, 16, kernel_size=9, stride=3, padding=4,
                      padding_mode="circular"),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=3, padding=2,
                      padding_mode="circular"),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(24),
            nn.Flatten(),
        )
        head_inputs = 32 * 24 + INTENTS + (2 if use_previous_command or use_relative_goal else 0)
        head_inputs += 16 if temporal else 0
        self.command_head = nn.Sequential(
            nn.Linear(head_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh(),
        )

        self.temporal_encoder = nn.GRU(12, 16, batch_first=True) if temporal else None

    def forward(self, scans, intent_ids, context):
        features = [self.scan_encoder(scans),
                    nn.functional.one_hot(intent_ids, self.num_intents).to(scans.dtype)]
        if self.use_previous_command or self.use_relative_goal:
            features.append(context)
        if self.temporal_encoder is not None:
            sectors = scans.reshape(scans.size(0), scans.size(1), 12, 30).amin(dim=-1)
            _, temporal_state = self.temporal_encoder(sectors)
            features.append(temporal_state.squeeze(0))
        return self.command_head(torch.cat(features, dim=1))


def load_goals(path):
    goals = {}
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            key = (row["family"], int(row["task"]), int(row["seed"]))
            goal = float(row["goal_x_m"]), float(row["goal_z_m"])
            if key in goals or not all(map(math.isfinite, goal)):
                raise ValueError(f"Invalid or repeated goal: {key}")
            goals[key] = goal
    return goals


def episode_tensors(path, goal=None, pilot_target=None):
    scans, commands, intents, relative_goals = [], [], [], []
    last_time = None
    target_columns = "waypoint" if pilot_target == "direction" else pilot_target
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        columns = {"time_s", "left_cmd", "right_cmd", "intent_id", "collision_count", *RANGE_COLUMNS}
        if pilot_target is not None:
            columns.update((f"{target_columns}_x_m", f"{target_columns}_z_m", "x_m", "z_m", "heading_deg"))
        if not columns.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path.name}: missing required columns")
        for row in reader:
            time = float(row["time_s"])
            if not math.isfinite(time) or (last_time is not None and
                    not math.isclose(time - last_time, PERIOD_S, abs_tol=0.002)):
                raise ValueError(f"{path.name}:{reader.line_num}: irregular scan timestamp")
            last_time = time
            intent = int(row["intent_id"])
            command = (float(row["left_cmd"]), float(row["right_cmd"]))
            scan = [float(row[name]) for name in RANGE_COLUMNS]
            if (intent not in range(INTENTS) or int(row["collision_count"]) != 0 or
                    not all(-1 <= value <= 1 for value in command) or
                    not all(value >= 0 for value in scan)):
                raise ValueError(f"{path.name}:{reader.line_num}: invalid intent, command or scan")

            target = goal
            scale = GOAL_SCALE_M
            if pilot_target is not None:
                target = float(row[f"{target_columns}_x_m"]), float(row[f"{target_columns}_z_m"])
                if not all(map(math.isfinite, target)):
                    raise ValueError(f"{path.name}:{reader.line_num}: invalid pilot target")
                if pilot_target in {"waypoint", "direction"}:
                    scale = 1.0
            if target is not None:
                dx = target[0] - float(row["x_m"])
                dz = target[1] - float(row["z_m"])
                heading = math.radians(float(row["heading_deg"]))
                cosine, sine = math.cos(heading), math.sin(heading)
                relative_goals.append(((-sine * dx + cosine * dz) / scale,
                                       (cosine * dx + sine * dz) / scale))
            intents.append(intent)
            commands.append(command)
            scans.append([min(value, RANGE_LIMIT_M) / RANGE_LIMIT_M for value in scan])

    if len(scans) < HISTORY:
        raise ValueError(f"{path.name}: fewer than {HISTORY} scans")
    scans = torch.tensor(scans, dtype=torch.float32)
    commands = torch.tensor(commands, dtype=torch.float32)
    goals = torch.tensor(relative_goals).clamp(-1, 1) if goal is not None or pilot_target is not None else None
    return scans, torch.tensor(intents), commands, goals


def split_recordings(directory, holdouts, versions=(8,), hall_only=False):
    matches = [(path, FILE_NAME.fullmatch(path.name)) for path in sorted(directory.glob("*.csv"))]
    partitions = [[], []]
    for path, match in matches:
        if match is None or int(match["version"]) not in versions:
            continue
        layout = match["layout"]
        if hall_only and not (layout.startswith("hall-") and path.name.startswith("completed-")):
            continue
        seed = re.search(r"-seed(\d+)$", layout)
        held_out = (any(layout == family or layout.startswith(f"{family}-") for family in holdouts)
                    or (seed is not None and int(seed[1]) % 5 == 4))
        partitions[int(held_out)].append(path)
    if not all(partitions):
        selected = ", ".join(f"v{version}" for version in versions)
        raise ValueError(f"Need recordings from {selected} in both training and held-out layouts")
    return partitions


PILOT_CASE = re.compile(r"-task(\d+)-seed(\d+)-(nominal|yaw)-")


def pilot_split(directory, mixed):
    recordings = {}
    for path in sorted(directory.glob("completed-teacher-v12-hall-*.csv")):
        match = PILOT_CASE.search(path.name)
        if match is None:
            continue
        mission, seed, condition = int(match[1]), int(match[2]), match[3]
        key = (seed, mission, "-clutter" in path.name)
        choices = recordings.setdefault(key, {})
        if condition in choices:
            raise ValueError(f"Duplicate pilot recording: {key}/{condition}")
        choices[condition] = path
    partitions = [[], []]
    for (seed, mission, cluttered), choices in sorted(recordings.items()):
        if set(choices) != {"nominal", "yaw"}:
            continue
        held_out = seed % 5 == 4
        use_yaw = mixed and not held_out and (seed + mission + cluttered) % 2 == 0
        partitions[int(held_out)].append(choices["yaw" if use_yaw else "nominal"])
    if not all(partitions):
        raise ValueError("Require paired nominal and yaw runs on train and validation layouts")
    return partitions


SIDE_SWAP_CASE = re.compile(r"-swap(left|right)-task00-seed(\d+)-")


def side_swap_split(directory):
    pairs = {}
    for path in sorted(directory.glob("completed-teacher-v12-sideswap-*.csv")):
        match = SIDE_SWAP_CASE.search(path.name)
        if match is None:
            raise ValueError(f"Unexpected side-swap recording: {path.name}")
        side, seed = match[1], int(match[2])
        sides = pairs.setdefault(seed, {})
        if side in sides:
            raise ValueError(f"Duplicate side-swap recording: {seed}/{side}")
        sides[side] = path
    partitions = [[], []]
    for seed, sides in sorted(pairs.items()):
        if set(sides) != {"left", "right"}:
            raise ValueError(f"Missing matched side-swap run for seed {seed}")
        partitions[int(seed % 7 == 6)].extend(sides[side] for side in ("left", "right"))
    if not all(partitions):
        raise ValueError("Require side-swap training and validation pairs")
    return partitions


TRANSFER_CASE = re.compile(r"^completed-teacher-v12-direction-[a-z-]+-follow-task(\d+)-seed(\d+)-")


def direction_transfer_split(directory):
    layouts = {}
    for path in sorted(directory.glob("completed-teacher-v12-direction-*.csv")):
        match = TRANSFER_CASE.match(path.name)
        if match is None:
            raise ValueError(f"Unexpected transfer recording: {path.name}")
        task, seed = int(match[1]), int(match[2])
        tasks = layouts.setdefault(seed, {})
        if task in tasks:
            raise ValueError(f"Duplicate transfer recording: {seed}/{task}")
        tasks[task] = path
    partitions = [[], []]
    for seed, tasks in sorted(layouts.items()):
        if set(tasks) != set(range(7)):
            raise ValueError(f"Incomplete seven-case transfer dataset for seed {seed}")
        partitions[int(seed % 5 == 4)].extend(tasks[task] for task in sorted(tasks))
    if not all(partitions):
        raise ValueError("Require transfer training and validation seeds")
    return partitions


RECOVERY_CASE = re.compile(r"^completed-teacher-v12-recovery-(normal|policy)-(?:staggered|angled-entry|wall-side)-follow-task(\d+)-seed(\d+)-")


def paired_recovery_paths(directory, source):
    cases = {}
    for path in sorted(directory.glob("completed-teacher-v12-recovery-*.csv")):
        match = RECOVERY_CASE.match(path.name)
        if match is None:
            raise ValueError(f"Unexpected recovery recording: {path.name}")
        kind, task, seed = match[1], int(match[2]), int(match[3])
        paths = cases.setdefault((seed, task), {})
        if kind in paths:
            raise ValueError(f"Duplicate recovery recording: {path.name}")
        paths[kind] = path
    return [paths[source] for (seed, _), paths in sorted(cases.items())
            if seed % 5 != 4 and set(paths) == {"normal", "policy"}]


def matched_recovery_windows(directory, source):
    paths = paired_recovery_paths(directory, source)
    counterpart = paired_recovery_paths(directory, "normal" if source == "policy" else "policy")
    if not paths or len(paths) != len(counterpart):
        raise ValueError("Expected complete matched recovery runs")

    def length(path):
        with path.open() as stream:
            next(stream)
            return sum(1 for _ in stream)

    indices = []
    lengths = []
    offset = 0
    for first, second in zip(paths, counterpart):
        if RECOVERY_CASE.match(first.name).groups()[1:] != RECOVERY_CASE.match(second.name).groups()[1:]:
            raise ValueError(f"Mismatched recovery recordings: {first.name} and {second.name}")
        count = length(first)
        selected = min(count, length(second))
        if selected < HISTORY:
            raise ValueError(f"Too few recovery scans in {first.name}")
        indices.append(torch.linspace(0, count - 1, selected).round().long() + offset)
        lengths.append(selected)
        offset += count
    print(f"matched {len(paths)} {source} restarts: {sum(lengths)} windows, per-run counts {lengths}")
    return paths, torch.cat(indices), offset


LARGE_CASE = re.compile(r"^completed-teacher-v12-direction-(?:open-furniture|scattered-islands|offset-partitions|variable-width-doors|offset-doors|post-door-clutter)-follow-task(\d+)-seed(\d+)-")


def large_arena_split(directory):
    recordings = {}
    for path in sorted(directory.glob("completed-teacher-v12-direction-*.csv")):
        match = LARGE_CASE.match(path.name)
        if match is None:
            raise ValueError(f"Unexpected large-arena recording: {path.name}")
        key = int(match[2]), int(match[1])
        if key in recordings:
            raise ValueError(f"Repeated large-arena recording for {key}")
        recordings[key] = path
    seeds = {seed for seed, _ in recordings}
    if len(seeds) < 2:
        raise ValueError("Large arena data requires separate training and validation seeds")
    validation_seed = max(seeds)
    partitions = [[], []]
    for (seed, _), path in sorted(recordings.items()):
        partitions[int(seed == validation_seed)].append(path)
    return partitions


DIRECTION_CHOICE_CASE = re.compile(
    r"^completed-teacher-v12-direction-choice-(island|fork)-(left|right)-seed(\d+)-")


def direction_choice_split(directory):
    pairs = {}
    for path in sorted(directory.glob("completed-teacher-v12-direction-choice-*.csv")):
        match = DIRECTION_CHOICE_CASE.match(path.name)
        if match is None:
            raise ValueError(f"Unexpected direction-choice recording: {path.name}")
        family, side, seed = match.groups()
        key = int(seed), family
        sides = pairs.setdefault(key, {})
        if side in sides:
            raise ValueError(f"Repeated direction-choice recording: {key}/{side}")
        sides[side] = path
    partitions = [[], []]
    for (seed, family), sides in sorted(pairs.items()):
        if set(sides) != {"left", "right"}:
            raise ValueError(f"Incomplete direction-choice pair: {seed}/{family}")
        partitions[int(seed % 5 == 4)].extend((sides["left"], sides["right"]))
    if not all(partitions):
        raise ValueError("Require complete direction-choice pairs in training and validation")
    return partitions


class ScanDataset(Dataset):
    def __init__(self, paths, goals=None, pilot_target=None, history_stride=1):
        def goal_for(path):
            match = HALL_CASE.search(path.name)
            if match is not None:
                key = (match[1], int(match[2]), int(match[3]))
                if key in goals:
                    return goals[key]
            raise ValueError(f"No Unity goal for {path.name}")

        episodes = [episode_tensors(path, goal_for(path) if goals is not None else None, pilot_target) for path in paths]
        self.scans, self.intents, self.commands = (torch.cat(values) for values in zip(*(item[:3] for item in episodes)))
        self.relative_goals = torch.cat([item[3] for item in episodes]) if goals is not None or pilot_target is not None else None
        starts = []
        offset = 0
        for scans, _, _, _ in episodes:
            starts.append(torch.full((len(scans),), offset, dtype=torch.long))
            offset += len(scans)
        self.starts = torch.cat(starts)
        self.history_offsets = torch.arange(HISTORY - 1, -1, -1) * history_stride
        self.previous = torch.zeros_like(self.commands)
        self.previous[1:] = self.commands[:-1]
        self.previous[self.starts.unique()] = 0

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, index):
        indices = (index - self.history_offsets).clamp_min(self.starts[index])
        context = self.previous[index] if self.relative_goals is None else self.relative_goals[index]
        return self.scans[indices], self.intents[index], context, self.commands[index]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    error = torch.zeros(INTENTS, device=device)
    count = torch.zeros(INTENTS, device=device)
    for scans, intents, previous, targets in loader:
        scans, intents, previous, targets = (x.to(device) for x in (scans, intents, previous, targets))
        sample_mae = (model(scans, intents, previous) - targets).abs().mean(dim=1)
        error.scatter_add_(0, intents, sample_mae)
        count.scatter_add_(0, intents, torch.ones_like(sample_mae))
    return (error.sum() / count.sum()).item(), {
        index: (error[index] / count[index]).item()
        for index in range(INTENTS) if count[index].item() > 0
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", type=Path, help="directory holding Unity teacher CSVs")
    parser.add_argument("--teacher-version", nargs="+", type=int, choices=(8, 9, 10, 11, 12), default=(8,))
    parser.add_argument("--output", type=Path, required=True, help="new TorchScript model path")
    parser.add_argument("--holdout", nargs="+", default=DEFAULT_HOLDOUTS, metavar="LAYOUT")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--no-previous-command", action="store_true", help="train the input ablation")
    parser.add_argument("--hall-only", action="store_true")
    parser.add_argument("--relative-goals", type=Path, help="Unity manifest for v12 hall goals")
    parser.add_argument("--pilot-target", choices=("goal", "waypoint", "direction"))
    parser.add_argument("--pilot-mixed-yaw", action="store_true")
    parser.add_argument("--pilot-blind-lidar", action="store_true")
    parser.add_argument("--direction-transfer-recordings", type=Path)
    parser.add_argument("--recovery-recordings", type=Path)
    parser.add_argument("--recovery-source", choices=("normal", "policy"))
    parser.add_argument("--matched-stress-recordings", type=Path, action="append", default=[])
    parser.add_argument("--matched-stress-source", choices=("normal", "policy"), action="append", default=[])
    parser.add_argument("--large-arena-recordings", type=Path, action="append", default=[])
    parser.add_argument("--direction-choice-recordings", type=Path)
    parser.add_argument("--direction-choice-fraction", type=float, default=1.0,
                        help="fraction of each direction-choice episode's windows used for training")
    parser.add_argument("--temporal-gru", action="store_true")
    parser.add_argument("--history-stride", type=int, choices=(1, 5), default=1)
    args = parser.parse_args()
    if (args.temporal_gru or args.large_arena_recordings or args.history_stride != 1 or
            args.direction_choice_recordings is not None) and args.pilot_target != "direction":
        parser.error("Large arena, direction-choice, and temporal pilots require --pilot-target direction")
    if not 0 < args.direction_choice_fraction <= 1 or (
            args.direction_choice_fraction != 1 and args.direction_choice_recordings is None):
        parser.error("Direction-choice fraction must be in (0, 1] and requires direction-choice recordings")
    if len(args.matched_stress_recordings) != len(args.matched_stress_source):
        parser.error("Provide both matched stress recovery directory and source")
    if args.matched_stress_recordings and args.recovery_recordings is None:
        parser.error("Matched stress training requires the original recovery dataset")
    if args.recovery_recordings is not None or args.recovery_source is not None:
        if args.direction_transfer_recordings is None or args.recovery_recordings is None or args.recovery_source is None:
            parser.error("Recovery training requires transfer recordings, recovery recordings and a recovery source")
    if args.direction_transfer_recordings is not None and args.pilot_target != "direction":
        parser.error("--direction-transfer-recordings requires --pilot-target direction")

    if args.pilot_target is not None and not (args.relative_goals is None and args.no_previous_command and
            args.teacher_version == [12] and (args.hall_only == (args.pilot_target != "direction"))):
        parser.error("--pilot-target requires v12 and no previous command, and --hall-only for hall targets")
    if args.pilot_blind_lidar and args.pilot_target not in {"waypoint", "direction"}:
        parser.error("--pilot-blind-lidar requires a waypoint or direction target")
    if args.pilot_mixed_yaw and args.pilot_target != "waypoint":
        parser.error("--pilot-mixed-yaw requires --pilot-target waypoint")

    if args.relative_goals is not None and not (args.hall_only and args.no_previous_command and args.teacher_version == [12]):
        parser.error("--relative-goals requires --hall-only --no-previous-command --teacher-version 12")
    if args.epochs < 1 or not args.recordings.is_dir() or args.output.exists():
        parser.error("provide a recordings directory, positive epoch count, and a new output path")
    try:
        matched_indices = []
        choice_episodes = None
        raw_stress_windows = 0
        if args.pilot_target == "direction":
            train_paths, val_paths = side_swap_split(args.recordings)
            if args.direction_transfer_recordings is not None:
                transfer_train, transfer_val = direction_transfer_split(args.direction_transfer_recordings)
                train_paths.extend(transfer_train)
                val_paths.extend(transfer_val)
            if args.recovery_recordings is not None:
                restarts = paired_recovery_paths(args.recovery_recordings, args.recovery_source)
                if not restarts:
                    raise ValueError("No matched completed recovery training runs")
                print(f"added {len(restarts)} paired {args.recovery_source} restarts")
                train_paths.extend(restarts)
            for directory in args.large_arena_recordings:
                large_train, large_val = large_arena_split(directory)
                if set(large_train + large_val).intersection(train_paths + val_paths):
                    raise ValueError(f"Repeated episodes in {directory}")
                print(f"large-arena data: {len(large_train)} training and {len(large_val)} validation runs")
                train_paths.extend(large_train)
                val_paths.extend(large_val)
            if args.direction_choice_recordings is not None:
                choice_train, choice_val = direction_choice_split(args.direction_choice_recordings)
                print(f"direction-choice data: {len(choice_train)} training and {len(choice_val)} validation runs")
                choice_start = len(train_paths)
                train_paths.extend(choice_train)
                choice_episodes = slice(choice_start, len(train_paths))
                val_paths.extend(choice_val)
            for directory, source in zip(args.matched_stress_recordings, args.matched_stress_source):
                stress_paths, indices, raw_windows = matched_recovery_windows(directory, source)
                if set(stress_paths).intersection(train_paths):
                    raise ValueError(f"Repeated episodes in {directory}")
                matched_indices.append(indices + raw_stress_windows)
                raw_stress_windows += raw_windows
                train_paths.extend(stress_paths)
        elif args.pilot_target is not None:
            train_paths, val_paths = pilot_split(args.recordings, args.pilot_mixed_yaw)
        else:
            train_paths, val_paths = split_recordings(args.recordings, args.holdout, args.teacher_version, args.hall_only)
        goals = load_goals(args.relative_goals) if args.relative_goals is not None else None
        train_set = ScanDataset(train_paths, goals, args.pilot_target, args.history_stride)
        val_set = ScanDataset(val_paths, goals, args.pilot_target, args.history_stride)
    except (ValueError, OSError) as error:
        parser.exit(2, f"{error}\n")

    if args.pilot_blind_lidar:
        train_set.scans.fill_(1.0)
        val_set.scans.fill_(1.0)

    torch.manual_seed(0)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.arange(len(train_set))
    if matched_indices:
        offset = len(train_set) - raw_stress_windows
        selected = torch.cat((torch.arange(offset), torch.cat(matched_indices) + offset))
    if choice_episodes is not None and args.direction_choice_fraction < 1:
        starts = train_set.starts.unique(sorted=True).tolist() + [len(train_set)]
        first, last = choice_episodes.start, choice_episodes.stop
        choice_windows = []
        for begin, end in zip(starts[first:last], starts[first + 1:last + 1]):
            count = max(1, round((end - begin) * args.direction_choice_fraction))
            choice_windows.append(torch.linspace(begin, end - 1, count).round().long())
        choice_indices = torch.cat(choice_windows)
        print(f"direction-choice windows: {len(choice_indices)}/{starts[last] - starts[first]}")
        selected = torch.cat((selected[selected < starts[first]], choice_indices,
                              selected[selected >= starts[last]]))
    training_data = train_set if len(selected) == len(train_set) else Subset(train_set, selected.tolist())
    train_loader = DataLoader(training_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=128)
    model = LowerPolicy(use_previous_command=not args.no_previous_command,
                        use_relative_goal=goals is not None or args.pilot_target is not None,
                        temporal=args.temporal_gru, history_stride=args.history_stride).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    best_loss = math.inf
    best_state = None

    print(f"scan history stride: {args.history_stride}")
    print(f"train: {len(train_paths)} runs, {len(training_data)} samples")
    print(f"validation: {len(val_paths)} runs, {len(val_set)} samples")
    if args.pilot_target is not None:
        print(f"yaw-disturbed training runs: {sum('-yaw-' in path.name for path in train_paths)}")
    for label, dataset in (("train", train_set), ("validation", val_set)):
        intents = dataset.intents[selected] if label == "train" else dataset.intents
        counts = torch.bincount(intents, minlength=INTENTS).tolist()
        print(f"{label} intent counts [follow, left, right, stop]: {counts}")
    persistence = (val_set.previous - val_set.commands).abs().mean().item()
    print(f"repeat-previous-command validation MAE {persistence:.4f}")
    for epoch in range(args.epochs):
        model.train()
        for scans, intents, previous, targets in train_loader:
            scans, intents, previous, targets = (x.to(device) for x in (scans, intents, previous, targets))
            loss = nn.functional.smooth_l1_loss(model(scans, intents, previous), targets, beta=0.1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_mae, by_intent = evaluate(model, val_loader, device)
        if val_mae < best_loss:
            best_loss = val_mae
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        print(f"epoch {epoch + 1:02d}: validation MAE {val_mae:.4f}, per intent {by_intent}")

    model.load_state_dict(best_state)
    model.cpu().eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(model)
    with args.output.open("xb") as file:
        torch.jit.save(scripted, file)
    print(f"saved {args.output}; best teacher-forced validation MAE {best_loss:.4f}")


if __name__ == "__main__":
    main()
