#!/bin/bash
#SBATCH --job-name=l2d-arm
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/work/23minhha/learn2design/logs/%x-%j.out
#SBATCH --error=/work/23minhha/learn2design/logs/%x-%j.err

# Run one experiment (optionally one arm of it) as a Slurm job.
# Usage (from ~/learn2design-2026 on the login node):
#   mkdir -p /work/23minhha/learn2design/logs
#   sbatch experiments/cluster/run_arm.sbatch 05_optimizer_race C0_raw_adam_K64
#   sbatch experiments/cluster/run_arm.sbatch 06_feasibility_tricks ""
#   # GPU variant: add --partition=gpu --gres=gpu:1 --time=04:00:00
set -euo pipefail
EXP=${1:?experiment dir, e.g. 05_optimizer_race}
export L2D_ARMS=${2:-}

cd "$HOME/learn2design-2026"
# dataset + upstream reference live at ~/upstream/Learn2Design-2026,
# symlinked into the repo as Learn2Design-2026/ (gitignored).
exec ~/.conda/envs/l2d/bin/python "experiments/${EXP}/run.py"
