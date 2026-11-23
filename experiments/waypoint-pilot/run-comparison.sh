#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")/../.." && pwd)
out="$root/experiments/waypoint-pilot"
work="$root/experiments/relative-goal-pilot/worktree"
unity='/media/airhood/storage/Unity/Editors/6000.3.27f1/Editor/Unity'
copy_a=/tmp/mini-lidar-unity-v9-paired.AsG0C7
copy_b=/tmp/mini-lidar-unity-v10-mixed.XJigDX

wait_for_collection() {
    local arm="$1" expected="$2" log="$out/$1/unity.log"
    for ((check=0; check<200; check++)); do
        if rg -q 'Teacher batch finished:' "$log"; then
            local actual
            actual=$(find "$out/$arm/recordings" -name '*.csv' | wc -l)
            [[ "$actual" == "$expected" ]]
            return
        fi
        if rg -q 'error CS|Aborting batchmode' "$log"; then
            tail -60 "$log" >&2
            return 1
        fi
        sleep 5
    done
    printf 'Timed out waiting for %s\n' "$arm" >&2
    return 1
}

train() {
    local arm="$1" target="$2"
    shift 2
    mkdir -p "$out/$arm/recordings"
    CUDA_VISIBLE_DEVICES='' python -u "$work/train_lower_policy.py" "$out/teacher/recordings" \
        --teacher-version 12 --hall-only --no-previous-command --epochs 15 \
        --pilot-target "$target" "$@" --output "$out/$arm/lower-policy.pt" > "$out/$arm/training.log" 2>&1
    python "$work/tools/export_unity_lower_policy.py" "$out/$arm/lower-policy.pt" "$out/$arm/LowerPolicy.bytes"
    printf 'Finished training %s\n' "$arm"
}

inference() {
    local arm="$1" project="$2"
    shift 2
    rsync -a "$work/mini_lidar_car-sim/Assets/Scripts/" "$project/Assets/Scripts/"
    local status=0
    "$unity" -batchmode -nographics -disableAudio -projectPath "$project" \
        -logFile "$out/$arm/unity.log" -executeMethod TeacherBatch.RunEvaluation \
        -teacherCollect -teacherInference -teacherHallsOnly -teacherWaypointPilot \
        -teacherFirstSeed 1128 -teacherSeedCount 2 -teacherPolicyName "$arm" \
        -teacherPolicyWeights "$out/$arm/LowerPolicy.bytes" -teacherOutput "$out/$arm/recordings" "$@" || status=$?
    if (( status != 0 )) && ! rg -q 'Teacher batch finished:' "$out/$arm/unity.log"; then
        tail -60 "$out/$arm/unity.log" >&2
        return "$status"
    fi
    wait_for_collection "$arm" 48
    printf 'Finished Unity rollouts %s\n' "$arm"
}

wait_for_collection teacher 96
train final-goal goal & final_train=$!
train waypoint waypoint & waypoint_train=$!
train waypoint-mixed waypoint --pilot-mixed-yaw & mixed_train=$!
wait "$final_train" "$waypoint_train" "$mixed_train"
wait_for_collection teacher-heldout 48
inference final-goal "$copy_a" & final_eval=$!
inference waypoint "$copy_b" -teacherWaypointInput & waypoint_eval=$!
wait "$final_eval"
inference waypoint-mixed "$copy_a" -teacherWaypointInput & mixed_eval=$!
wait "$waypoint_eval" "$mixed_eval"
python "$work/tools/summarize_waypoint_pilot.py" "$out"
