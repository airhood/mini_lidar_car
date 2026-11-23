#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
run=experiments/unity-v8-seeds-1000-1119
manifest="$run/training-files.txt"
host="${LIDAR_TRAINING_SSH_HOST:?Set an SSH user and host for your training server}"
remote=mini_lidar_car-runs/v8-20261119
ssh_opts=(-i "$HOME/.ssh/id_ed25519" -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=8)
rsync_shell='ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=8'

test -s "$manifest"
test -s "$run/final-audit.json"

ssh "${ssh_opts[@]}" "$host" 'bash -s' <<'REMOTE'
set -euo pipefail
dir="$HOME/mini_lidar_car-runs/v8-20261119"
gpu=$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader,nounits)
test -n "$gpu"
nvidia-smi -i 1 --query-gpu=index,name,memory.free --format=csv,noheader
if [[ -e "$dir/launch.pid" || -e "$dir/lower-policy.pt" ]]; then
    printf '%s\n' "Training run already exists at $dir. Check training.log." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$gpu"
for candidate in "$HOME/miniconda3/envs/openvla-ivm/bin/python" \
                 python3 "$HOME/miniconda3/bin/python" "$HOME/anaconda3/bin/python"; do
    if python=$(command -v "$candidate") &&
        "$python" -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
        mkdir -p "$dir/data"
        printf '%s\n' "$gpu" > "$dir/gpu-uuid"
        printf '%s\n' "$python" > "$dir/python-path"
        exit 0
    fi
done
echo 'CUDA-enabled PyTorch could not be loaded with GPU 1.' >&2
exit 1
REMOTE

rsync -rz --partial --files-from="$manifest" -e "$rsync_shell" \
    data/drive-recordings/ "$host:$remote/data/"
rsync -rz -e "$rsync_shell" train_lower_policy.py "$manifest" "$host:$remote/"

ssh "${ssh_opts[@]}" "$host" 'bash -s' <<'REMOTE'
set -euo pipefail
cd "$HOME/mini_lidar_car-runs/v8-20261119"
test ! -e launch.pid
test ! -e lower-policy.pt
while IFS= read -r name; do
    test -s "data/$name"
done < training-files.txt
export CUDA_VISIBLE_DEVICES="$(cat gpu-uuid)"
python=$(cat python-path)
nohup "$python" -u train_lower_policy.py data --output lower-policy.pt \
    > training.log 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > launch.pid
sleep 2
kill -0 "$pid" || { tail -25 training.log; exit 1; }
printf 'Training started on physical GPU 1 (%s), PID %s\n' "$CUDA_VISIBLE_DEVICES" "$pid"
printf 'Training log: %s/training.log\n' "$PWD"
REMOTE
