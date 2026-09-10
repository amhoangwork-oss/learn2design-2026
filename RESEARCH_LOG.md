# RESEARCH_LOG — learn2design-2026

| Exp | Question | Status |
|---|---|---|
| 01 | Exact loss structure, param inventory, CPU cost | ✅ |
| 02 | Gradient conditioning per property family | ✅ |
| 03 | Dataset reproducibility (dataset → problem → params) | ✅ |
| 04 | 1-D response probes (log-power, log(1−R), periodicity) | ✅ |
| 05 | Optimizer horse-race on seed 42 (Adam variants, preconditioning) | next |
| 06 | Feasibility repair: penalty schedule + best-feasible tracking | next |

---

## Exp 01 — Loss structure, parameter inventory, CPU timing (2026-09-09)

**Setup.** `UIFOProblem(size=3, topology_seed=42)` (topology `DBAHHHEGG-SLHLSSSLSLLL`,
homodyne), CPU-only, dfbench 0.3.3 / differometor 0.0.5 / jax 0.9.0.1.
Script: `experiments/01_loss_structure/run.py` → `results/01_loss_structure/summary.json`.

**Results.**

| quantity | value |
|---|---|
| n_params | 187 (4 coupled pairs: vertical/horizontal spaces share one value) |
| property counts | reflectivity 52, tuning 52, mass 51, length 16, power 6, db 5, angle 5 |
| loss at midpoint | 6.842 = sensitivity 4.861 + penalty 1.981 |
| feasible at midpoint | no |
| grad norm / absmax | 16.6 / 15.8 |
| problem init | 49 s |
| JIT compile (CPU, new trace) | **935 s** |
| compiled value / value_and_grad | **0.15 ms / 0.12 ms** |

**Interpretation.** Objective cost is tracing-dominated; fix one trace per run and
batch with `vmap` at fixed size. Raw bounds span ~12 orders of magnitude. Feasibility
repair is a first-class subproblem (midpoint infeasible). Loss verified in source:
`L = mean_f log10(S/S_voy) + Σ_j squash_relu(P_j/T_j)`; feasibility is a separate hard
max-check; score = min loss over feasible logged evals (random-search fallback if none).

---

## Exp 02+03 — Dataset reproduction + gradient conditioning (2026-09-09)

**Setup.** `dataset.h5`: 29,650 entries (28,863 size-3, 787 size-4), 12,437 unique
topologies; largest group 2,878 entries (`DABFEEGBB-SLSLLSLSSDLS`, best −0.3648).
Rebuilt that problem and evaluated its 8 best saved param vectors.

**Results.**

1. **Reproduction is exact**: max |saved − recomputed| = 4.7e−6 over 8 best entries;
   all feasible, penalty 0. Dataset→problem→params pipeline is sound.
2. **0/29,650 entries match topology seed 42** — exact-match seeding cannot be the
   main strategy; cross-topology transfer can (per-property statistics, init models).
3. **Gradient conditioning at a dataset optimum** (bounded space, median |grad|):
   reflectivity 2.7e2 · tuning 2.6e−2 · power 1.3e−3 · length 2.1e−5 ·
   mass 1.3e−10 · db 3.6e−12 · angle 8.5e−17 → **19 orders of magnitude** across
   families (within-family max/median up to 2e14).
4. Dataset per-family value spreads (normalized std 0.21–0.48) are a usable metric
   prior; reflectivity median 0.72, mass median ~197/200 (wall-hugging), power median 12.

**Interpretation.** Raw-space gradient descent is hopeless without per-coordinate
scaling; Adam's RMS normalization is the only reason NAAdamGD works at all, and its
plateau at ~0.5 is consistent with noise re-breaking the scaling. Dataset optima are
wall-constrained in reflectivity (|∇| ≈ 0 everywhere except R-family) — log(1−R)
coordinates convert that wall into a soft direction.

---

## Exp 04 — 1-D response probes per property family (2026-09-10)

**Setup.** Seed-42 topology, midpoint start; probe single coordinates ±2/5/10/25%
of span per family; tuning periodicity test L(v) vs L(v−360) at 3 coordinates.
Script: `experiments/04_1d_probes/run.py` → `results/04_1d_probes/summary.json`.

**Results (loss range over the probe, all at fixed other coords).**

| family | probe range (abs) | Δ loss | shape observed |
|---|---|---|---|
| reflectivity (idx175, base 0.5) | 0.25→0.75 | **0.89** | sharp valley at ~0.48 (7.18), walls on both sides; highly non-convex in R |
| tuning (idx146, base 0) | −180→180 | **1.33** | broad peak at 0±36°, minima at ±180° (6.842) — response is *phase-like* with structure at 90° scale |
| power (idx165, base 100) | 50→150 | 0.31 | monotone increasing in raw W over the window (6.73@50 → 7.05@150): lower power = better loss here (penalty fixed!) |
| length (idx150, base 2000) | 1000→3000 | 0.18 | smooth, monotone decreasing over window; gentle |
| db (idx148, base 5) | 2.5→7.5 | 0.001 | flat |
| mass (idx180, base 100) | 50→150 | 0.000 | flat |
| angle (idx126, base 0) | −90→90 | 0.000 | flat at midpoint config |

**Periodicity test.** L(300°) vs L(−60°): equal to 1e−10 at two coordinates, but
**differs by 4.7e−2 at idx 1** — one component (a squeezer-adjacent tuning, seed-42
idx1) is NOT 360-periodic in its effect (its tuned length/phase couples through
physical space, not just phase). Periodicity holds for most, not all, tuning coords.

**Interpretation (the reparameterization story).**

1. **Reflectivity is the dominant non-convexity.** Δ=0.89 loss across ±25% R with a
   narrow valley — in raw R space this is a needle. Cavity physics says the natural
   coordinate is round-trip gain `u = log(1−R)`: the valley at R≈0.48 becomes a
   smooth exponential-ridge; walls at R→1 (dataset optima sit there) become soft.
   → R2 reparameterization confirmed as necessary.
2. **Power enters the loss ≈ linearly in log-power** over the probed window
   (ΔL/Δlog₁₀P ≈ 0.31/0.48 ≈ 0.65 per decade): sensitivity ~ P^(−0.3)-ish in this
   regime — log-power coordinates flatten this to a linear ridge with mild constant
   curvature. → R1 confirmed.
3. **Tuning response is phase-structured at 90° scale** (quarter-wave): raw [-360,360]
   boxes hide that only phase *differences* matter; the ±180° minima vs 0° peak shows
   sign-flip symmetry (π-periodic for mirror tunings, as expected for reflection off
   both sides). A sin/cos embedding captures the wrap; the idx-1 counterexample says:
   don't force global periodicity — let the optimizer keep ±360 box but initialize
   near 0/180 equivalences. → R3 revised: use wrap-aware *initialization* + periodic
   basin perturbations, not a hard reparameterization.
4. **mass/db/angle are locally flat at generic points** — their gradients only come
   alive near structured configurations. In a multi-basin scheme they contribute
   ~nothing early; per-coordinate Adam rescaling handles them without special-casing,
   but their *dataset-informed* priors (mass hugging 200!) matter for initialization.

**Consequence for optimizer design.** Combine:
- coordinates: `x = [log(1−R) per mirror; log(P) per laser; length/√span; mass/σ_m;
  tuning·π/180 as-is; angle·π/180 as-is]` with per-family static scaling from dataset
  stds (R4), inside the *algorithm* (unbounded mode with custom unit mapping);
- basins: dataset entries (same-topology if available, else nearest-complexity
  topologies' optima as warm starts) + jittered copies;
- optimizers: Adam-family with static preconditioner + decaying noise (NAAdam
  pattern) for basin descent; L-BFGS warm restarts for polish; penalty schedule
  relu→squashed; track best-feasible separately at every step.

---

## Environment notes

- conda env `~/miniconda3/envs/learn2design` (python 3.12), `pip install -e ".[cuda12]"`;
  CPU-only via `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu`.
- GPU busy (vLLM/ASR); all experiments CPU-only.
- CPU timing: new-trace JIT ≈ 5–15 min; compiled eval ≈ 0.1–0.2 ms; vmap traces
  compile separately — one vmap trace per experiment, fixed batch size.
- dfbench: result-producing methods require `start_logging()`; `warmup_*()` before.
  Fresh `Objective(max_time=...)` per script keeps the clock honest.
- Exp 04 total runtime ≈ 6.5 h CPU for 7×8 probes + 3 periodicity pairs (mostly
  3 vmap-free single evals each — the aux trace recompiles once per new call family).
