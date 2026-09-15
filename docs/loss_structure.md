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

## 7. The exact sim chain (source-verified, differometor 0.0.5 on cluster)

Everything below is batched over the 50 frequencies (HLO `f64[.,50,~702,~701]`);
`solve` = `jnp.linalg.solve`, one linear solve per system.

1. **Carrier solve** (`simulate.py:290`): `M_c(θ,f) x_c = b`. Matrix entries written
   per component (`components.py`):
   - space: `φ = −exp(−i·2πf·L·n/c)` (line 174)
   - laser: `E = sqrt(2P/ε₀c)·e^{i·phase}` (line 232)
   - mirror/BS: reflectivity fractions, tuning phases `e^{i·tuning}`
   - squeezer: `cosh(2r)` / `sinh(2r)·e^{2i·α}` quadrature mixing (line 283)
2. **Signal solve** (`simulate.py:298–338`): connector entries scaled by the carrier
   field, upper/lower sideband blocks forced conjugate-symmetric, `x_s = M_s^{-1} b_s`.
3. **Quantum noise via adjoint chain** (`simulate.py:340–412`):
   selections `s = W_qhd ⊙ x_c` (QHD phases scattered in), then
   `w = M_s^{-H} s`, `v = N(θ) w`, `C = M_s^{-1} v`, and
   `q² = 2·(UNIT_VACUUM·h·F₀/4)·Re⟨s, C⟩`
   **= const · Re[ s^H M_s^{-1} N M_s^{-H} s ]** — a Hermitian quadratic form in the
   carrier field vector, sandwiching the noise-source covariance through the inverse
   signal transfer matrix.
4. **Sensitivity** (`utils.py:346`): `S(f) = sqrt(q² + (4e-9·P₁)² + (1e-8·f·P₂)²)/P₀`
   with `P₀ = |conj(x_c)·x_{s,up} + x_c·x_{s,low}|` at the homodyne detectors
   (`components.py:829`); P₁/P₂ = signal proxies for amplitude/frequency noise.
5. **Powers** (`utils.py:126`): `P_port = 0.5·ε₀c·|E_port|²` (`components.py:809`);
   hard/soft side = per-component **max** over its ports; isolators → soft side.
   Carrier is a single (DC) solve — port powers are f-independent.
6. **Feasibility** (`dfbench/problems/base_problem.py:186`): the hard max check
   `max_j P_j/T_j ≤ 1` elementwise over hard (3.5e6) / soft (2e3) / detector (1e-2).
7. **Loss** (`base_problem.py:240`): `mean_f log10(S/S_voy) + Σ_j p(P_j/T_j)`.

## 8. Boundary structure — exact homogeneity in the joint power scale (derived)

Carrier fields are linear in each laser amplitude `sqrt(P_k)`:
`E_j = Σ_k sqrt(P_k)·G_jk(θ_optics)` with optics-only gains `G_jk`. Under a joint
rescale `P_k → t·P_k` (all lasers, one scalar t):
`E_j → sqrt(t)·E_j ⇒ P_j → t·P_j` — **exact**, since `|sqrt(t)·x|² = t|x|²`.

Sensitivity scalings under the same rescale (noise proxies ride the carrier):
`q² ∝ t` (shot noise — the quadratic form is quadratic in the carrier), signal and
classical-noise proxies `∝ t` (modulation sources scale with carrier amplitude —
the standard laser-referred convention; consistent with Exp 04's probe showing the
loss decreasing monotonically in power over decades, ≈0.65 loss-decades per
log-decade). Hence for every frequency:

```
S(t)² = (α/t + β + γf²) / δ²        α,β,γ,δ > 0, optics-dependent
```

**strictly decreasing in t, saturating at the classical floor `sqrt(β+γf²)/δ`.**

Consequences:
1. **The optimum always sits ON the feasibility boundary.** For any optics, the
   best feasible power scale is `t* = min_j T_j/P_j(θ)` — the max-constraint is
   active at any optimum (whichever port saturates first — often the detector,
   capped at 10 mW, which directly caps `P_sig`).
2. **Feasibility restoration is closed-form.** One aux eval returns all `P_j`;
   the largest feasible joint rescale is `t* = min(1, min_j T_j/P_j)`. Cost ≈ one
   eval (~0.15 ms). No line search, no extra solves.
3. **There are exactly two improvement channels**, coupled through
   `S* = sqrt(α/t* + β + γf²)/δ`: (a) grow headroom `t*` (min-max power objective),
   (b) shrink the noise coefficients α, β, γ (cavity/isolation design).

## 9. Constrained reformulation — proposed Exp 10

The problem is a smooth NLP: `min mean_f log10(S/S_voy) s.t. P_j(θ) ≤ T_j, θ ∈ box`
(only chosen nonsmoothness: the relu/squashed penalty and the port maxima, which
are locally smooth away from ties). Published-technique options:

- **C1 interior point / log-barrier**: add `−μ Σ_j log(1 − P_j/T_j)`, schedule μ↓.
  Every iterate strictly feasible ⇒ **every logged eval counts for the score**
  (today most evals are infeasible and discarded by the scorer). First-order
  friendly (Adam or L-BFGS on the augmented objective). Use log-space constraints
  `c_j = log(P_j/T_j)` — by §8 homogeneity `∂c_j/∂log t = 1` exactly, so the
  constraint-side conditioning (19 orders in raw space) collapses to O(1).
- **C2 augmented Lagrangian**: `f + Σ_j [λ_j c_j + (ρ/2)c_j²]₊` with dual ascent
  `λ ← max(λ + ρc, 0)`. Multiplier memory prevents the in/out-of-feasibility
  oscillation that plain penalties show (stronger version of Exp 06's premise).
- **C3 feasible-projected Adam**: run the true loss (zero penalty) and after every
  step apply the closed-form §8.2 restoration, logging the projected point.
  Projection cost ≈ 1 eval/step — negligible at 0.15 ms.
- **C4 trust-constr / filter-SQP polish**: accept steps if objective *or* violation
  improves (no penalty parameter); natural replacement for the plain L-BFGS stage.

**Equivalent / auxiliary objectives:**
- **A1 gauge fixing (exact equivalent)**: pin the joint power scale at the boundary
  `t*(θ_rest)` and optimize only the remaining coords, unconstrained. Every eval
  feasible by construction; the 6 power coords collapse to 1 boundary variable.
- **A2 headroom min-max**: minimize `softmax_β max_j log(P_j/T_j)` (β annealed ↑)
  as a first phase — a pure "make room" objective — then spend it (true loss,
  power pinned at the new boundary).
- **A3 conditional laser optimum (QCQP)**: at fixed optics, laser amplitudes u enter
  every power as `|G_j u|²` and the readout as `|G₀u|²`. Maximizing detector power
  s.t. hard/soft caps is a convex QCQP (generalized Rayleigh quotient) — globally
  solvable via generalized eigenvectors/SDP. Since `∂log S/∂log P_sig = −1`
  uniformly (steepest global direction, §2.2), this makes the 6 power coords (and
  their constraint coupling) an analytic step instead of a search dimension.
  (Check first whether laser phases are competition-exposed; that extends A3.)
- **A4 worst-band softmin pretraining**: auxiliary weighted loss emphasizing the
  worst frequency bands to find better basins, then polish on the true mean.
