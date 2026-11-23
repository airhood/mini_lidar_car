#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")/../.." && pwd)
out="$root/experiments/side-swap-pilot"
work="$root/experiments/relative-goal-pilot/worktree"
unity='/media/airhood/storage/Unity/Editors/6000.3.27f1/Editor/Unity'
copy_a=/tmp/mini-lidar-unity-v9-paired.AsG0C7
copy_b=/tmp/mini-lidar-unity-v10-mixed.XJigDX

train() {
    local arm="$1"
    shift
    mkdir -p "$out/$arm/recordings"
    CUDA_VISIBLE_DEVICES='' python -u "$work/train_lower_policy.py" "$out/teacher/recordings" \
        --teacher-version 12 --no-previous-command --epochs 15 --pilot-target direction \
        "$@" --output "$out/$arm/lower-policy.pt" > "$out/$arm/training.log" 2>&1
    python "$work/tools/export_unity_lower_policy.py" "$out/$arm/lower-policy.pt" "$out/$arm/LowerPolicy.bytes"
}

evaluate() {
    local arm="$1" project="$2"
    shift 2
    local status=0
    "$unity" -batchmode -nographics -disableAudio -projectPath "$project" \
        -logFile "$out/$arm/unity.log" -executeMethod TeacherBatch.RunEvaluation \
        -teacherCollect -teacherInference -teacherSideSwap \
        -teacherFirstSeed 2550 -teacherSeedCount 10 -teacherPolicyName "$arm" \
        -teacherPolicyWeights "$out/$arm/LowerPolicy.bytes" -teacherOutput "$out/$arm/recordings" "$@" || status=$?
    if (( status != 0 )) && ! rg -q 'Teacher batch finished:' "$out/$arm/unity.log"; then
        tail -60 "$out/$arm/unity.log" >&2
        return "$status"
    fi
}

for ((i=0; i<120; i++)); do
    if rg -q 'Teacher batch finished:' "$out/teacher/unity.log" && rg -q 'Teacher batch finished:' "$out/teacher-heldout/unity.log"; then
        break
    fi
    sleep 3
done
python "$work/tools/summarize_side_swap.py" "$out" --teachers-only
train lidar & lidar_train=$!
train blind --pilot-blind-lidar & blind_train=$!
wait "$lidar_train"
wait "$blind_train"
evaluate lidar "$copy_a" & lidar_eval=$!
evaluate blind "$copy_b" -teacherWaypointBlind & blind_eval=$!
wait "$lidar_eval"
wait "$blind_eval"
python "$work/tools/summarize_side_swap.py" "$out"
