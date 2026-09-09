# Learn2Design-2026 — UIFO detector design optimization

**Repo:** https://github.com/amhoangwork-oss/learn2design-2026
**Status:** Phase 1 — loss-structure analysis & optimizer design (CPU only; GPU reserved).

## Current state (update this file whenever strategy changes)

- Competition: optimize ~200 continuous params of a hidden UIFO topology, 4 h wall-clock,
  score = mean over 10 topologies of best **feasible** loss.
- Loss (verified in `dfbench/src/dfbench/problems/base_problem.py::_calculate_loss`):

  ```
  L(θ) = mean_f log10( S(θ, f) / S_voyager(f) ) + Σ_j p(P_j(θ) / T_j)
  p(x) = max(x-1, 0) / (1 + max(x-1, 0))     [squashed ReLU, default]
  feasibility: P_hard ≤ 3.5e6, P_soft ≤ 2e3, P_detector ≤ 1e-2   (hard maxima, independent of p)
  ```

- Round-1 leaderboard: top = 0.020, 2nd = 0.071, organizer best baseline (NAAdamGD) = 0.504.
  Negative losses are possible; every run has a random-search fallback score, so there is a
  floor on how badly a run can go.

## Strategy (blueprint)

1. **Loss-structure analysis** (Exp 01–03): parameter → loss-factor mapping, gradient
   conditioning, feasibility geometry.
2. **Reparameterization experiments** (Exp 04+): log-power space for laser `power`,
   periodic coordinate for `tuning`/`angle`, softplus parameter for `reflectivity`,
   per-property metric preconditioning (Adam-style, but static — from dataset statistics).
3. **Multi-basin pipeline**: dataset-seeded basin starts → batched Adam variants with
   decaying noise → warm-restart L-BFGS polish per basin → feasibility repair via
   penalty schedule (squashed → relu) → keep best feasible.
4. Submit as `submission.py` (single `OptimizationAlgorithm` subclass + requirements.txt).

## Environment

- conda env: `~/miniconda3/envs/learn2design` (Python 3.12, jax 0.9.0.1 CPU+cuda12 pkg, dfbench 0.3.3, differometor 0.0.5)
- Upstream clones (read-only reference, gitignored): `Learn2Design-2026/` (incl. dfbench source + competition_data/round1 + dataset.h5), `Learn2Design-2026/differometor_src/`
- CPU-only runs: `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu` (GPU reserved for ML training)

## Key files

| Path | What |
|---|---|
| `RESEARCH_LOG.md` | Chronological experiment results |
| `experiments/` | Numbered experiment scripts (each writes `results/NN_*/) |
| `docs/loss_structure.md` | Loss equations + reparameterization derivations |

## Rules we operate under

- Feasibility (`is_feasible`) is a hard max-check and is **independent** of the penalty
  function — penalty tricks only shape the optimization landscape, never the score.
- `best_loss` may be infeasible; the scorer takes the best feasible eval. Always log
  `is_feasible` and track best-feasible separately.
- Evaluation = wall-clock budget. CPU evals ≈ 10× slower than H100 ⇒ scale short experiments
  accordingly; final timing tests need GPU later.
