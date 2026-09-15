#!/bin/bash
#SBATCH --job-name=l2d-arm
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/work/23minhha/learn2design/logs/%x-%j.out
#SBATCH --error=/work/23minhha/learn2design/logs/%x-%j.err

# Run one experiment (optionally one arm of it) as a Slurm job.
# NOTE: per-user job cap = 6 — for MULTIPLE arms prefer run_matrix.sbatch
# (one job, all GPUs, arms in waves); use this template only for single runs.
# Usage (from ~/learn2design-2026 on the login node):
#   mkdir -p /work/23minhha/learn2design/logs
#   sbatch experiments/cluster/run_arm.sbatch 05_optimizer_race C0_raw_adam_K64
#   sbatch experiments/cluster/run_arm.sbatch 06_feasibility_tricks ""
#   # GPU variant (preferred when idle A5000s — see sinfo -p gpu):
#   L2D_DEVICE=gpu L2D_CHUNK=64 sbatch -J l2d-05-P0 -p gpu --gres=gpu:1 -t 02:00:00 \
#     experiments/cluster/run_arm.sbatch 05_optimizer_race P0_prec_adam_K64
set -euo pipefail
EXP=${1:?experiment dir, e.g. 05_optimizer_race}
export L2D_ARMS=${2:-}

cd "$HOME/learn2design-2026"
# dataset + upstream reference live at ~/upstream/Learn2Design-2026,
# symlinked into the repo as Learn2Design-2026/ (gitignored).
exec ~/.conda/envs/l2d/bin/python "experiments/${EXP}/run.py"
