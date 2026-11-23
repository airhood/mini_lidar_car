#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")/../.." && pwd)
out="$root/experiments/mid-route-recovery-pilot"
work="$root/experiments/relative-goal-pilot/worktree"
transfer="$root/experiments/direction-transfer-pilot"
original="$root/experiments/side-swap-pilot"
copy=/tmp/mini-lidar-unity-v10-mixed.XJigDX
unity='/media/airhood/storage/Unity/Editors/6000.3.27f1/Editor/Unity'
export PYTHONPATH="$work"

train() {
  local arm="$1"
  mkdir -p "$out/$arm-model"
  CUDA_VISIBLE_DEVICES='' python -u "$work/train_lower_policy.py" "$original/teacher/recordings" \
    --teacher-version 12 --no-previous-command --pilot-target direction --epochs 15 \
    --direction-transfer-recordings "$transfer/teacher-development/recordings" \
    --recovery-recordings "$out/restart-teacher/recordings" --recovery-source "$arm" \
    --output "$out/$arm-model/lower-policy.pt" > "$out/$arm-model/training.log" 2>&1
  python "$work/tools/export_unity_lower_policy.py" "$out/$arm-model/lower-policy.pt" "$out/$arm-model/LowerPolicy.bytes"
}

rollout() {
  local arm="$1" seed="$2" scene="$3"
  shift 3
  mkdir -p "$out/$arm/recordings"
  local status=0
  "$unity" -batchmode -nographics -disableAudio -projectPath "$copy" \
    -logFile "$out/$arm/unity.log" -executeMethod "$scene" -teacherCollect \
    -teacherDirectionTransfer -teacherStaggeredOnly -teacherFirstSeed "$seed" \
    -teacherSeedCount 5 -teacherOutput "$out/$arm/recordings" "$@" || status=$?
  if (( status != 0 )); then rg -q 'Teacher batch finished:' "$out/$arm/unity.log"; fi
  rg 'Teacher batch finished:' "$out/$arm/unity.log"
}

model_rollout() {
  local arm="$1" cohort="$2" seed="$3" weights="$4"
  rollout "$arm-$cohort" "$seed" TeacherBatch.RunEvaluation -teacherInference \
    -teacherPolicyName "$arm" -teacherPolicyWeights "$weights"
}

python "$work/tools/inspect_mid_route_recovery.py" "$out"
train normal & normal_pid=$!
train policy & policy_pid=$!
rollout teacher-new 3400 TeacherBatch.Run
model_rollout baseline new 3400 "$transfer/mixed-lidar/LowerPolicy.bytes"
wait "$normal_pid"
wait "$policy_pid"
for arm in normal policy; do
  model_rollout "$arm" previous 3100 "$out/$arm-model/LowerPolicy.bytes"
  model_rollout "$arm" new 3400 "$out/$arm-model/LowerPolicy.bytes"
done
MPLCONFIGDIR=/tmp/mini-lidar-matplotlib MPLBACKEND=Agg python "$work/tools/inspect_mid_route_recovery.py" "$out" --transfer "$transfer"
