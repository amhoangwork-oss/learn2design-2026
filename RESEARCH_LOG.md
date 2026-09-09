# RESEARCH_LOG — learn2design-2026

Reverse-chronological within each experiment; newest experiments at the bottom.
Each entry: motivation → setup → result → interpretation → next action.

---

## Exp 01 — Loss structure, parameter inventory, CPU timing (2026-09-09)

**Motivation.** Understand the exact objective and its computational profile before
touching optimizers. The user hypothesis: the loss encodes reparameterizations that
simplify the non-convex space.

**Setup.** `UIFOProblem(size=3, topology_seed=42)`, CPU-only
(`CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu`), dfbench 0.3.3 / differometor 0.0.5 /
jax 0.9.0.1. Midpoint params, `value` + `value_and_grad` + `value_aux`.
Script: `experiments/01_loss_structure/run.py` → `results/01_loss_structure/summary.json`.

**Results.**

| quantity | value |
|---|---|
| topology string | `DBAHHHEGG-SLHLSSSLSLLL` (homodyne) |
| n_params | 187 (4 coupled = vertical/horizontal space pairs share one value) |
| property counts | reflectivity 52, tuning 52, mass 51, length 16, power 6, db 5, angle 5 |
| loss at midpoint | 6.842 = sensitivity 4.861 + penalty 1.981 |
| feasible at midpoint | no |
| grad norm / absmax | 16.6 / 15.8 |
| problem init | 49 s (Voyager reference + build) |
| JIT compile, first `value()` call | **935 s** (CPU, 30 cores) |
| compiled `value()` | **0.15 ms** |
| compiled `value_and_grad()` | **0.12 ms** (warm from `warmup_value_and_grad()`) |

**Interpretation.**

1. The objective after compilation is *tiny* on modern hardware — the cost is JIT
   tracing, not evaluation. Consequence for algorithm design: fix ONE batch shape and
   ONE eval family per run; never vary shapes mid-run; do exploration with `vmap`
   batches of fixed size. Compile count per run should be O(1)–O(5).
2. Round-1 winners averaged ~10 evals/s on H100 (leaderboard `mean_evaluations_per_second`),
   i.e. their Python loop + logging dominates, exactly as our 0.15 ms raw eval suggests.
   A tight loop can afford ~10^5–10^6 evals within 4 h — population methods are cheap.
3. Property inventory shows two periodic families (tuning, angle: 57/187 params) and
   two scale-heavy families (mass [0.01,200], length [0.1,4000], power [0,200]).
   Raw bound ranges span 12 orders of magnitude overall ⇒ per-property metric (R4)
   is mandatory, not optional.
4. Midpoint is infeasible with penalty ≈ 2 (≈ at least two fully-violating elements,
   since squashed penalty saturates at 1/element). Feasibility repair is a real
   subproblem, not an edge case.
5. Loss confirmed (source-verified, `docs/loss_structure.md`):
   `L = mean_f log10(S/S_voy) + Σ_j squash_relu(P_j/T_j)`; feasibility is a separate
   hard max-check; score = min loss over feasible evals.

**Next.** Exp 02 (gradient conditioning across properties at dataset points),
Exp 03 (reproduce dataset entries' saved losses — validates the dataset pipeline),
Exp 04 (1-D finite-difference probes per property group → test R1/R2/R3 curvature claims).

---

## Environment notes

- conda env `~/miniconda3/envs/learn2design` (python 3.12). Installed with
  `pip install -e ".[cuda12]"` from the upstream clone; CPU-only runs via
  `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu` (GPU reserved; JAX falls back with a
  noisy but harmless CUDA plugin error).
- GPU is busy (vLLM / ASR eval). All experiments in this log are CPU-only.
- JIT compile ≈ 15 min/new trace on CPU. All experiment scripts must reuse a single
  compiled trace per run and batch with `vmap` at fixed batch size.
