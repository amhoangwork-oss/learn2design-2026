# Learn2Design-2026 — UIFO detector design optimization

**Repo:** https://github.com/amhoangwork-oss/learn2design-2026
**Status:** Phase 1 complete (loss-structure analysis). Phase 2 scaffolds (Exp 05–07)
written, not yet run. Compute moved to the **hpc-cei HPC cluster**; workflow:
code here → push GitHub → `git pull` on cluster → sbatch.

## Current state (update whenever strategy changes)

- Competition: optimize ~200 continuous params of a hidden UIFO topology, 4 h
  wall-clock, score = mean over 10 topologies of best **feasible** loss. Round-1
  winner: 0.020; organizer best baseline NAAdamGD: 0.504.
- Deadline: final submission **15 Oct 2026 AoE** (public leaderboard rounds 26 Aug /
  12 Sep / 29 Sep 2026). Upstream/starter kit: `artificial-scientist-lab/Learn2Design-2026`.
- Loss (source-verified):
  `L = mean_f log10(S/S_voy) + Σ_j p(P_j/T_j)`, `p = squash-relu`, thresholds
  hard 3.5e6 / soft 2e3 / detector 1e−2; feasibility is a separate hard max-check;
  score = min loss over feasible logged evals (random-search fallback if none).
- Parameter families per size-3 topology (n≈187): reflectivity 52, tuning 52,
  mass 51, length 16, power 6, db 5, angle 5; 4 coupled pairs.
- Execution: this Windows checkout is **code + docs only** (no dataset, no env).
  All runs happen on the hpc-cei cluster (see Environment).

## Verified findings (see RESEARCH_LOG.md for details)

1. Eval cost after JIT ≈ 0.15 ms CPU — tracing dominates; fix one trace per run.
2. Gradient conditioning across families ≈ **19 orders of magnitude** (raw space).
3. Dataset (29,650 designs) reproduces exactly (4.7e−6); 0 entries match a random
   topology ⇒ cross-topology transfer, not lookup.
4. 1-D probes: reflectivity valley Δ0.89 (needle in raw space), power ≈ linear in
   log-power (0.65/decade), tuning phase-structured at 90°, mass/db/angle locally
   flat at generic points; tuning periodicity holds for most but not all coords.

## Strategy (blueprint)

1. **Coordinates** (inside our algorithm, unbounded mode + custom unit mapping):
   - reflectivity → `u = log(1−R)` (softens the R→1 wall where dataset optima live)
   - power → `log P` (linearizes the response, 0.65 loss-decades per log-decade)
   - static per-family scaling from dataset stds (diagonal preconditioner)
   - tuning/angle: keep bounded box, wrap-aware initialization & perturbations
2. **Multi-basin pipeline**: dataset-warm starts + jitter → batched Adam (optax)
   with static preconditioner + NAAdam-style decaying noise → keep top basins →
   L-BFGS warm-restart polish → best-feasible tracked separately at every step.
3. **Feasibility**: penalty schedule relu (crash) → squashed (sit at wall); never
   rely on penalty=0 for scoring; always log `is_feasible`.
4. Submit as single `OptimizationAlgorithm` subclass + requirements.txt (evaluator
   has no network; bundle everything).

## Phase 2 experiment plan (on the cluster)

Order: **07 → (05 ∥ 06) → 08 → 09**. One sbatch per arm — arms run
wall-clock-parallel instead of Phase 1's 6.5 h sequential CPU runs. Each experiment
writes `results/NN_name/`; commit the JSON summaries back to this repo and log the
numbers in RESEARCH_LOG.md.

| Exp | Question | Cluster shape |
|---|---|---|
| 07_k_scaling | vmap batch ceiling K on A5000 (fp32 vs fp64 cost); H100 transfer via TFLOP/bw ratios + safety margin (CPU K-scaling dropped) | 1–2 GPU sbatch jobs — run first, sets K for 09 |
| 05_optimizer_race | raw vs preconditioned Adam × noise on/off, K=64/256 (seed 42) | one GPU sbatch per arm (idle A5000s, wall-clock-parallel) |
| 06_feasibility_tricks | zero-penalty phase → squashed repair vs squashed-only | one GPU sbatch per arm (A/B/C) |
| 08_multitopo_transfer | best config on ~10 held-out dataset topologies (the actual score shape: mean over topologies) | job array over topologies |
| 09_end2end_4h | full pipeline (basins → Adam → L-BFGS → best-feasible) under competition-like 4 h budget, logged | single job on A5000 + CPU comparison; submission dry-run |

**GPU-first rule.** Check `sinfo -p gpu` for idle A5000s before submitting; run arms
one-GPU-per-job (`sbatch -p gpu --gres=gpu:1`) instead of CPU nodes, so the whole
arm matrix iterates in wall-clock parallel. CPU nodes are the fallback only.

**H100 transfer rule (K-scaling without CPU runs).** Measured on the first GPU
attempt: the UIFO sim is **float64 natively** (f64[.,50,~700,~700] propagation
tensors, ~200 MB/sample forward) regardless of `jax_enable_x64`, and unchunked
vmap OOMs on a 24 GB A5000 at K=32. So Exp 07 sweeps **chunk sizes** (Exp 05/06
loop chunks to reach total K), not K. Transfer is bandwidth-bound: A5000
768 GB/s → H100 PCIe 2039 / SXM 3352 GB/s = **2.65× / 4.36× nominal**; with a
0.5× safety margin the A5000-optimal chunk is the guaranteed-safe floor and
~1.3–2.2× larger is likely. (FP64 compute ratio 60–78× bounds the
compute-bound case from above.) Confirm at first H100 access; don't burn
CPU-hours re-measuring on CPU nodes.

## Environment

**Local (Windows checkout)** — code + docs only; no dataset, no Python env, no
local execution. Edit here → push GitHub → pull on cluster.

**hpc-cei cluster** (ssh alias `hpc-cei`, user 23minhha; details in the
hpc-cei-cluster skill):
- Login `hpc-gw.local`; Slurm; partitions `compute` (48-core CPU nodes) + `gpu`
  (4× RTX A5000 24 GB each, cc 8.6). Compute nodes have no internet — pip/staging
  on the login node only.
- Code: `~/learn2design-2026` (git pull to update). Upstream reference clone +
  dataset.h5: `~/upstream/Learn2Design-2026` (gitignored in our repo; dataset
  symlinked where run.py expects it).
- Env: conda `l2d` (`module load python/miniforge3; conda activate l2d`) —
  jax[cuda12], optax, dfbench 0.3.3, differometor 0.0.5, h5py.
- Job data + Slurm logs: `/work/23minhha/learn2design/` (60-day idle purge — copy
  result JSONs into the repo promptly).
- Phase-1 reference (previous machine): conda env `~/miniconda3/envs/learn2design`;
  CPU timing: new-trace JIT ≈ 5–15 min; compiled eval ≈ 0.15 ms; vmap traces compile
  separately (fix batch size per run).

## Key files

| Path | What |
|---|---|
| `RESEARCH_LOG.md` | Experiment results (chronological, with numbers) |
| `docs/loss_structure.md` | Loss equations + reparameterization derivations |
| `experiments/NN_name/run.py` | Numbered experiments; each writes `results/NN_name/` |

## Rules we operate under

- Feasibility is independent of the penalty function — penalty tricks only shape
  the search.
- `best_loss` may be infeasible; track best-feasible separately.
- All compute on hpc-cei **via Slurm only** (never the login node). A5000s are fair
  game for batched optimization now; the H100 eval env remains the timing referee
  (A5000 numbers are a proxy).
- GPU reserved for ML training locally — local machine runs no experiments.
