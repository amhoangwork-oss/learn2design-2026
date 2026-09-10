# Learn2Design-2026 — UIFO detector design optimization

**Repo:** https://github.com/amhoangwork-oss/learn2design-2026
**Status:** Phase 1 complete (loss-structure analysis). Phase 2: optimizer horse-race.

## Current state (update whenever strategy changes)

- Competition: optimize ~200 continuous params of a hidden UIFO topology, 4 h
  wall-clock, score = mean over 10 topologies of best **feasible** loss. Round-1
  winner: 0.020; organizer best baseline NAAdamGD: 0.504.
- Loss (source-verified):
  `L = mean_f log10(S/S_voy) + Σ_j p(P_j/T_j)`, `p = squash-relu`, thresholds
  hard 3.5e6 / soft 2e3 / detector 1e−2; feasibility is a separate hard max-check;
  score = min loss over feasible logged evals (random-search fallback if none).
- Parameter families per size-3 topology (n≈187): reflectivity 52, tuning 52,
  mass 51, length 16, power 6, db 5, angle 5; 4 coupled pairs.

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

## Environment

- conda env: `~/miniconda3/envs/learn2design` (Python 3.12, jax 0.9.0.1, dfbench 0.3.3,
  differometor 0.0.5). CPU-only runs: `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu`.
- Upstream clones (read-only, gitignored): `Learn2Design-2026/` (dfbench source +
  competition_data/round1 + dataset.h5), `Learn2Design-2026/differometor_src/`.
- CPU timing: new-trace JIT ≈ 5–15 min; compiled eval ≈ 0.15 ms; vmap traces compile
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
- GPU reserved for ML training — experiments CPU-only for now; final timing tests
  on GPU later (H100 eval env).
