# RESEARCH_LOG — learn2design-2026

Reverse-chronological within each phase; newest at the bottom of the table.
Each entry: motivation → setup → result → interpretation → next action.

| Exp | Question | Status |
|---|---|---|
| 01 | Exact loss structure, param inventory, CPU cost | ✅ done |
| 02 | Gradient conditioning per property family | ✅ done |
| 03 | Dataset reproducibility (dataset → problem → params) | ✅ done |
| 04 | 1-D response probes (log-power, log(1−R), periodicity) | ⏳ running |
| 05 | vmap batch scaling + multi-restart Adam vs NAAdamGD on seed-42 | next |
| 06 | Feasibility repair: relu→squashed penalty behavior | next |

---

## Exp 01 — Loss structure, parameter inventory, CPU timing (2026-09-09)

**Motivation.** Understand the exact objective and its computational profile before
touching optimizers. Hypothesis (user): the loss encodes reparameterizations that
simplify the non-convex space.

**Setup.** `UIFOProblem(size=3, topology_seed=42)`, CPU-only, dfbench 0.3.3 /
differometor 0.0.5 / jax 0.9.0.1. Midpoint params, value + value_and_grad + aux.
Script: `experiments/01_loss_structure/run.py` → `results/01_loss_structure/summary.json`.

**Results.**

| quantity | value |
|---|---|
| topology string | `DBAHHHEGG-SLHLSSSLSLLL` (homodyne) |
| n_params | 187 (4 coupled pairs: vertical/horizontal spaces share one value) |
| property counts | reflectivity 52, tuning 52, mass 51, length 16, power 6, db 5, angle 5 |
| loss at midpoint | 6.842 = sensitivity 4.861 + penalty 1.981 |
| feasible at midpoint | no |
| grad norm / absmax | 16.6 / 15.8 |
| problem init | 49 s (Voyager reference + build) |
| JIT compile (CPU, first value) | **935 s** |
| compiled value / value_and_grad | **0.15 ms / 0.12 ms** |

**Interpretation.**

1. Objective cost after compile is negligible — the cost is *tracing*. Fix one batch
   shape and one eval family per run; compile count must be O(1)–O(5).
2. Property inventory: two periodic families (tuning+angle = 57/187), two scale-heavy
   families (mass [0.01,200], length [0.1,4000], power [0,200]); raw bounds span ~12
   orders of magnitude.
3. Midpoint is infeasible (penalty ≈ 2 ≈ ≥2 saturated elements) — feasibility repair
   is a first-class subproblem.
4. Loss (source-verified, `docs/loss_structure.md`):
   `L = mean_f log10(S/S_voy) + Σ_j squash_relu(P_j/T_j)`; feasibility is a separate
   hard max-check; score = min loss over feasible logged evals (random-search fallback
   if none).

---

## Exp 02+03 — Dataset reproduction + gradient conditioning (2026-09-09)

**Setup.** `dataset.h5`: 29,650 entries (28,863 size-3, 787 size-4),
12,437 unique topologies; largest topology group has 2,878 entries
(`DABFEEGBB-SLSLLSLSSDLS`, best loss −0.3648). Rebuilt its problem with
`UIFOProblem(size=3, topology=...)`, evaluated the 8 best saved param vectors.

**Results.**

1. **Reproduction is exact**: max |saved − recomputed| = **4.7e−6** over the 8 best
   entries (float32 vs float64 roundoff); all feasible, penalty 0. The dataset→
   problem→params pipeline is sound. Dataset points are directly usable as basins.
2. **0/29,650 entries match topology seed 42** — dataset coverage of the topology
   space is sparse. Exact-match seeding cannot be the main strategy; cross-topology
   transfer (per-property statistics, learned initializers) can.
3. **Gradient conditioning at a dataset optimum is extreme** (bounded space):

   | property | median |grad| | absmax |grad| |
   |---|---|---|
   | reflectivity | 2.7e+02 | 1.8e+10 |
   | tuning | 2.6e−02 | 5.2e+03 |
   | power | 1.3e−03 | 5.2e+02 |
   | length | 2.1e−05 | 5.6e−02 |
   | mass | 1.3e−10 | 1.6e+01 |
   | db | 3.6e−12 | 4.8e−11 |
   | angle | 8.5e−17 | 2.0e−15 |

   Spread across families ≈ **19 orders of magnitude** (max/med within a family ≈ 2e14).
   This is the single strongest confirmation of the reparameterization hypothesis:
   raw-space gradient descent is doomed to crawl along reflectivity walls while
   ignoring angle/mass entirely. Per-property (even per-coordinate) diagonal
   preconditioning is mandatory.
4. **Dataset value statistics** (2,878 entries of one topology): normalized stds
   0.21–0.48 per family — dataset spread is a usable per-property scale prior.
   reflectivity median 0.72, mass median ~197 (near upper bound 200), power median 12.

**Interpretation for the optimizer.**

- Adam's per-coordinate normalization is not just helpful here — it is the *only*
  thing that makes the raw problem trainable; but its bias toward early gradients on
  a 19-decades-mismatched landscape explains why NAAdamGD plateaus at ~0.5: noise
  rescales as raw steps, re-breaking the scaling.
- Better: *static* per-coordinate metric from the dataset (σ of values per family),
  combined with log-space for power/reflectivity, then Adam/L-BFGS on top.
- The dataset optimum's |∇| ≈ 0 in every family except reflectivity (2.7e2 median)
  suggests the saved "optima" are wall-constrained in reflectivity (R→1 boundary),
  consistent with the reflectivity=1−eps bound guard. Optimizing in
  `u = log(1−R)` converts that wall into a soft direction.

**Next.** Exp 04 probes (running), then Exp 05 optimizer horse-race on seed 42.

---

## Environment notes

- conda env `~/miniconda3/envs/learn2design` (python 3.12). `pip install -e ".[cuda12]"`
  from the upstream clone; CPU-only runs via `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu`.
- GPU is busy (vLLM/ASR). All experiments so far are CPU-only.
- CPU timing: new-trace JIT ≈ 5–15 min; compiled eval ≈ 0.1–0.2 ms. Batched (vmap)
  traces compile separately — budget one vmap trace per experiment, batch size fixed.
- dfbench gotcha: every result-producing method (value/value_aux/grad) requires
  `start_logging()` first; use `warmup_*()` before that. Budget a fresh
  `Objective(max_time=...)` per script to keep the clock honest.
