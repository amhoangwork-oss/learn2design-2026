#!/bin/bash
#SBATCH --job-name=l2d-matrix
#SBATCH --partition=gpu
#SBATCH --nodes=1-4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
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
# all its arms sequentially in one process. Override GPUs/nodes on the CLI:
#   --gres=gpu:4              (max 4 GPUs/node; more arms -> automatic waves)
#   --nodes=2 --gres=gpu:4    (MULTI-NODE: one arm per GPU on every node, all
#                              arms start at once via srun placement — up to
#                              4*Nnodes arms simultaneously in ONE job slot)
# NOTE --nodes>1 needs the srun branch below: plain background processes only
# see the batch node's GPUs; Slurm places tasks via srun, nothing automatic.

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

# ---------------- MULTI-NODE path (--nodes=N, N>1) ----------------
# srun places each arm on a specific node of the allocation (background
# processes would only see the batch node's GPUs). One arm per GPU per node,
# all arms start at once. Requires --gres=gpu:4 on every node (4 slots/node).
if [ "$SLURM_JOB_NUM_NODES" -gt 1 ]; then
  mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
  echo "multi-node allocation: ${NODES[*]} (4 arm slots per node)"
  i=0
  for node in "${NODES[@]}"; do
    for g in 0 1 2 3; do
      [ "$i" -ge "$N" ] && break 2
      read -r exp arm <<< "${PAIRS[$i]}"
      if [ "$arm" = "ALL" ]; then arm_arg=""; else arm_arg="$arm"; fi
      tag="${exp}-${arm}"
      echo "[launch-mn] exp=$exp arm=$arm node=$node gpu=$g -> $LOGD/${tag}-${SLURM_JOB_ID}.out"
      srun --nodes=1 --ntasks=1 --overlap -w "$node" \
        env L2D_DEVICE=gpu L2D_ARMS="$arm_arg" CUDA_VISIBLE_DEVICES="$g" \
        "$PY" "experiments/$exp/run.py" \
        > "$LOGD/${tag}-${SLURM_JOB_ID}.out" 2>&1 &
      wave_pids+=("$!")
      i=$((i + 1))
    done
  done
  for p in "${wave_pids[@]}"; do wait "$p" || fail=1; done
  echo "matrix finished multi-node (fail=$fail)"
  exit "$fail"
fi

# ---------------- single-node path (unchanged) ----------------
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
