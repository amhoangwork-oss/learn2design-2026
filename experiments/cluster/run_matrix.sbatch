#!/bin/bash
#SBATCH --job-name=l2d-matrix
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --output=/work/23minhha/learn2design/logs/%x-%j.out
#SBATCH --error=/work/23minhha/learn2design/logs/%x-%j.err

# ONE Slurm job for a whole arm matrix — per-user job cap is 6, so never spray
# one job per arm. Requests all GPUs as a single allocation and drives one arm
# process per GPU (waves when there are more arms than GPUs).
#
# Usage (from ~/learn2design-2026 on the login node):
#   sbatch experiments/cluster/run_matrix.sbatch \
#     "05_optimizer_race:C0_raw_adam_K64,P0_prec_adam_K64 06_feasibility_tricks:A_squashed_always"
# Spec: space-separated "exp:arm1,arm2" entries; a bare "exp" (no colon) runs
# all its arms sequentially in one process. Override GPUs/walltime on the CLI:
#   --gres=gpu:4 -t 08:00:00   (max 4 GPUs/node; more arms -> automatic waves)

set -euo pipefail
SPEC=${1:?spec like "05_optimizer_race:C0_raw_adam_K64,P1_prec_noise_K64 06_feasibility_tricks:A_squashed_always"}

cd "$HOME/learn2design-2026"
PY="$HOME/.conda/envs/l2d/bin/python"
LOGD=/work/23minhha/learn2design/logs
mkdir -p "$LOGD"

# GPUs of THIS allocation (Slurm sets CUDA_VISIBLE_DEVICES under cgroup isolation)
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -ra DEVS <<< "$CUDA_VISIBLE_DEVICES"
else
  mapfile -t DEVS < <(nvidia-smi -L | sed -n 's/^GPU \([0-9]*\):.*/\1/p')
fi
NGPU=${#DEVS[@]}
echo "allocation GPUs (${NGPU}): ${DEVS[*]}"

# Flatten spec -> "exp arm" pairs
PAIRS=()
for entry in $SPEC; do
  case "$entry" in
    *:*) exp=${entry%%:*}; arms=${entry#*:} ;;
    *)   exp=$entry; arms="" ;;
  esac
  if [ -z "$arms" ]; then
    PAIRS+=("$exp ALL")
  else
    IFS=',' read -ra AL <<< "$arms"
    for a in "${AL[@]}"; do PAIRS+=("$exp $a"); done
  fi
done
N=${#PAIRS[@]}
echo "arm processes to run: $N"

fail=0
i=0
while [ "$i" -lt "$N" ]; do
  wave_pids=()
  for ((g = 0; g < NGPU && i < N; g++, i++)); do
    read -r exp arm <<< "${PAIRS[$i]}"
    if [ "$arm" = "ALL" ]; then arm_arg=""; else arm_arg="$arm"; fi
    tag="${exp}-${arm}"
    echo "[launch] exp=$exp arm=$arm gpu=${DEVS[$g]} -> $LOGD/${tag}-${SLURM_JOB_ID}.out"
    L2D_DEVICE=gpu L2D_ARMS="$arm_arg" CUDA_VISIBLE_DEVICES="${DEVS[$g]}" \
      "$PY" "experiments/${exp}/run.py" \
      > "$LOGD/${tag}-${SLURM_JOB_ID}.out" 2>&1 &
    wave_pids+=("$!")
  done
  for p in "${wave_pids[@]}"; do wait "$p" || fail=1; done
  echo "[wave] done (${#wave_pids[@]} arms)"
done
echo "matrix finished (fail=$fail)"
exit "$fail"
