# UIFO loss structure & reparameterization notes

All claims below verified against source:
- `dfbench/src/dfbench/problems/base_problem.py` (`_calculate_loss`, penalty presets, thresholds)
- `dfbench/src/dfbench/problems/uifo/uifo_problem.py` (bounds, parameter list)
- `differometor/utils.py` (`calculate_sensitivities`, `sensitivity_qamplfreq_noise`, `calculate_powers`, `sigmoid_bounding`)
- `differometor/components.py` (`power_detector`, `signal_detector`, thresholds)

## 1. The exact loss

```
L(θ) = mean_{f ∈ F} log10( S(θ,f) / S_voy(f) )  +  Σ_j p(P_j(θ)/T_j)
p(x) = relu(x - 1) / (1 + relu(x - 1))       # squashed ReLU, per-element
```

- `F` = 50 log-spaced frequencies, 20 → 5000 Hz.
- `S(θ,f) = sqrt(q² + a² + φ²) / P_sig` (quantum + amplitude + frequency noise,
  normalized by detector signal power), computed from **three** parallel simulations
  (space / amplitude / frequency modulation) sharing the same θ.
- Penalties per element: hard-side component powers `P` vs `T_hard = 3.5e6`,
  soft-side vs `T_soft = 2e3`, detector vs `T_det = 1e-2`.
- **Scoring feasibility is separate and hard**: `is_feasible = (all P ≤ their T)`.
  The penalty only shapes the search; it never changes the score. The submitted run
  score is `min loss over logged evals with is_feasible=True` (random-search fallback
  if none).

## 2. Immediate structural observations

1. **The loss is a mean of per-frequency log-ratios.** Each `log10(S/S_ref)` term is
   dominated by the worst frequency band; the mean makes all 50 points matter.
   Improving any single frequency point improves the loss — there is no competition
   between bands in the objective itself (it is a plain mean, not soft-min or max).
2. **Noise terms enter squared inside a sqrt; the signal enters linearly in the
   denominator.** `∂L/∂log P_sig = -1/log(10) × mean(1)` — the detector signal power is
   the one parameter that shifts *all* frequencies at once, i.e. the steepest global
   direction. Laser `power` (bounded [0, 200]) is the direct knob for it — but it is
   exactly the knob that saturates hard/soft power constraints.
3. **Powers are quadratic in fields; fields are rational in parameters.** The
   resonant enhancement of a Fabry–Perot-like cavity is
   `P_cav ∝ P_in / (1 - r₁ r₂)²` in power (field `∝ 1/(1-r₁r₂)`). The loss is *not*
   quadratic in `(1-r₁r₂)`: log-sensitivity is (to first order) linear in
   `log(1/(1-r₁r₂))`-like quantities ⇒ **log-space reparameterization of round-trip
   gains is the natural metric**.
4. **Feasibility is one-sided (max power ≤ threshold).** The feasible set is
   *downward closed* in every power: scaling all laser powers down keeps feasibility.
   A feasible point exists at any topology (pump everything off ⇒ detector power → 0
   but sensitivity loss explodes; the random-search fallback proves feasible points
   are found trivially). The real task: push sensitivity while keeping headsroom.
5. **The squashed penalty saturates at 1 per element** (bounded in [0,1)).
   `n_constraints` per topology = #(hard group) + #(soft group) + #(detector).
   With many violating components the penalty is nearly flat ⇒ weak gradient signal
   ⇒ the *relu* preset (linear overshoot) is the better "crash through the wall"
   driver early, squashed is the better "sit at the wall" driver late. A
   **penalty schedule relu → squashed** is promising (this is allowed:
   `set_penalty_fn` before `start_logging`... but note: only *one* mode per run after
   logging starts; so schedule must be implemented by the algorithm itself by tracking
   both aux losses — see Exp 03).
6. **tuning & angle are periodic in [-360, 360)**. Both bounds map to the *same*
   physical phase (mod 360). The sigmoid-bounded box treats ±360 as hard walls ⇒ the
   optimizer must not want to cross 0 phase; but physically, +359.9° and -0.1° are
   adjacent. A **sin/cos embedding** (2 coords per phase) removes the artificial
   wall; equivalently, initialize `tuning` near 0 deg or reparametrize
   `tuning = 360·atan2(y, x)/π`. Caveat: for a *fixed* algorithm contract (params in
   bounded space), we can only exploit this in our own internal representation —
   the mapping into bounded space must stay within bounds (wrap around, not clip).

## 3. Parameter inventory (verified for a size-3 topology, seed 42)

Property bounds (defaults):

| property | range | physical meaning | notes |
|---|---|---|---|
| reflectivity | [~0, ~1] | mirror R | bounded off 0/1 by eps=1e-12 to avoid NaN grads |
| tuning | [-360, 360] | mirror/BS phase | periodic! |
| db | [0, 10] | isolator loss dB | one-sided |
| angle | [-360, 360] | squeezer angle | periodic! |
| power | [0, 200] | laser power W | enters powers *linearly*, sensitivity ~ sqrt |
| mass | [0.01, 200] | test mass | suspension/thermal noise |
| length | [0.1, 4000] | arm/space length | enters phase as 2πL/λ · N — periodic in λ-ish scale! |

Coupled parameters: vertical/horizontal space pairs share one value
(`constrain_inter_grid_cell_spaces`) — the optimizer sees one coordinate, the setup
writes it to multiple edges. Exp 01 counts these exactly.

## 4. Proposed reparameterizations (to be validated in Exp 02/04)

Goal: make each coordinate's *loss-vs-coordinate* profile look like a well-scaled
quadratic bowl in the regime the optimizer cares about, and remove artificial walls.

R1. **log-power for laser `power`:** p = log(P/P₀), P = P₀ e^p clamped to [0,200].
    Rationale: sensitivity noise terms ∝ powers linearly (amplitude, frequency noise
    ∝ signal powers) but detector signal enters as denominator; power appears
    multiplicatively through a linear chain: log is the natural metric; also
    the *feasible* range typically spans decades.
R2. **log-space on (1 − R) and round-trip gain products:** for each mirror,
    u = log(1 − R). Cavity enhancement is rational in (1 − R) ⇒ log-space linearizes
    the dominant response. Reflectivity bound [0,1] with eps guard ⇒ u ∈
    [log(1−1+1e-12), log(1)] = [−27.6, −0] — a huge dynamic range in u, so *standardize*
    per property using dataset statistics (see R4).
R3. **sin/cos for tuning and angle** (period 360): replace the box with a circle.
    In our internal parameterization; map back with wrap.
R4. **per-property affine metric (diagonal preconditioner) from the 30k dataset:**
    compute, per property type, the std of optimized values across 29,650 designs;
    scale each coordinate by 1/σ_prop. This approximates the ideal local metric far
    better than the raw ranges (which span [1e-12, 200] in scale).
R5. **soft-side power headroom as an explicit penalty schedule**: optimization in two
    phases: (a) maximize signal using penalty=relu (crash against constraints),
    (b) polish with squashed penalty + L-BFGS from the best feasible point.
    Both phases track `is_feasible` at every eval and record best-feasible separately.

## 5. Optimizer design (draft, to be filled after experiments)

- Basins: sample K inits from dataset entries matching the hidden topology when
  possible (topology string match); else random + dataset-continuation pretraining.
- Coarse: batched Adam (optax) with R1–R4 preconditioning, decaying noise
  (NAAdam-style), ~50–100 basins, early discard.
- Fine: warm-restart L-BFGS (optax lbfgs) per surviving basin, cosine restarts.
- Feasibility: track best-feasible; never rely on penalty=0.

## 6. Open questions (Exps 02–05 answer these)

- Q1: Is the sigmoid-bounded space better than bounded-space + clipping for Adam?
- Q2: How big is the gradient conditioning spread across properties (Exp 02)?
- Q3: Does log-power actually flatten the sensitivity response? (Exp 04 finite-diff probe)
- Q4: Are dataset entries' losses reproducible with the competition problem?
  (Exp 03: evaluate 10 saved entries; validates our whole dataset-based pipeline.)
