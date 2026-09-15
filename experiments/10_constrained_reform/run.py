"""Exp 10: constrained reformulation — boundary methods vs penalty descent.

Background (docs/loss_structure.md §7–9, source-verified): port powers are EXACTLY
homogeneous of degree 1 in the joint laser-power scale t (fields linear in sqrt(P)),
and the sensitivity S(t)^2 = (alpha/t + beta + gamma f^2)/delta^2 is strictly
decreasing in t. Therefore every optimum sits ON the feasibility boundary, and the
boundary rescale t* = safety*exp(-max_j c_j) (c_j = log(P_j/T_j), capped by the
[0, P_MAX] power box) is available in closed form from one aux eval — pushing
power DOWN when infeasible and UP to spend headroom when the optics allow it.
Exp 10 replaces the penalty layer with boundary-aware methods while keeping the
Exp 05/06 optimizer (batched Adam, uniform base config) fixed.

Uniform base config: base_config.json (K, lr, noise, transform, restarts...).
Set it from the Exp 05/06 winner BEFORE submitting; arms differ only in their
feasibility layer. All arms share topology/pool/seeds; metrics: best-feasible
sensitivity loss (the score), feasible-eval fraction, time-to-first-feasible,
mean headroom t*, box-cap fraction.

Arms (kind, what it isolates):
  D0_control      config-driven penalty phases (default squashed; set to the
                  05/06 winner schedule) — control, no projection
  P1_proj_state   zero penalty; every eval self-projects the joint power scale to
                  the feasibility boundary (inside the jitted graph, gradient
                  flows through the projection); logged point = projected point
  P2_gauge_fixed  P1 + power-coordinate grads and noise masked (relative laser
                  split frozen at init; joint scale set purely by projection)
  PB_proj_logonly zero-penalty descent on the raw point (no constraint feedback);
                  logs the projected boundary point — separates "good logged
                  points" from "good trajectory"
  B1_barrier      one-time boundary projection at init, then log-barrier phases
                  mu = 1e-1 -> 1e-2 -> 1e-3 (fresh trace per phase); iterates
                  strictly interior by construction
  L1_alm          augmented Lagrangian on log-space constraints c_j = log(P_j/T_j):
                  sens + sum_j [relu(lam+rho c)^2 - lam^2]/(2rho), per-basin dual
                  ascent lam <- relu(lam + rho c) every dual_every steps
  H1_headroom     phase 1 (headroom_frac): minimize softmax_beta max_j log(P_j/T_j)
                  with power scale frozen (pure optics "make room" objective);
                  phase 2: P1-style projected descent ("spend it")

NOTE on stale-trace hazard: set_penalty_fn REBINDS problem.objective_function_aux
(a fresh jitted closure per build); penalty fns are baked at trace time. All
objective access therefore goes through OF() which resolves the problem attribute
at call time — never hold a module-level reference (this exact bug invalidated
Exp 06 arms B/C in job 1198; fixed there too).

Writes results/10_constrained_reform/summary_<arm>.json (checkpoints every 25
steps) + summary.json. Env knobs: L2D_ARMS (comma list), L2D_CHUNK, L2D_DEVICE,
L2D_SMOKE=1 (K=8, 2 steps/phase — fast scaffold validation).
"""
import json
import os
import time

import h5py
import numpy as np

if os.environ.get("L2D_DEVICE") == "gpu":
    os.environ.pop("JAX_PLATFORMS", None)  # use the sbatch-allocated GPU
else:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import optax

from dfbench import Objective
from dfbench.problems import UIFOProblem
from dfbench.problems.base_problem import (
    squashed_relu_penalty,
    relu_penalty,
    zero_penalty,
    HARD_SIDE_POWER_THRESHOLD,
    SOFT_SIDE_POWER_THRESHOLD,
    DETECTOR_POWER_THRESHOLD,
)

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "..", "results", "10_constrained_reform")
os.makedirs(OUT, exist_ok=True)

# ---------------- uniform base config (set from Exp 05/06 winner) ----------------
with open(os.path.join(HERE, "base_config.json")) as f:
    CFG = json.load(f)
SMOKE = os.environ.get("L2D_SMOKE") == "1"
if SMOKE:
    CFG = dict(CFG, K=8, noise0=0.0, warm_restart_every=None, dual_every=2)
    print("SMOKE MODE: K=8, 2 steps/phase", flush=True)

TOPO = "DBAHHHEGG-SLHLSSSLSLLL"
TOPO_DS = "DABFEEGBB-SLSLLSLSSDLS"
H5 = os.path.join(HERE, "..", "..", "Learn2Design-2026", "dataset", "dataset.h5")
FAMS = ["reflectivity", "tuning", "mass", "length", "power", "db", "angle"]

# ---------------- dataset warm-start pool (same machinery as Exp 05/06) ----------------
h5 = h5py.File(H5, "r")
entries = h5["entries"][:]
topo = np.array([t.decode() if isinstance(t, bytes) else t for t in entries["topology_string"]])
idx = np.where(topo == TOPO_DS)[0]
order = idx[np.argsort(np.array(entries["loss"])[idx])]
_ds_problem = UIFOProblem(size=3, topology=TOPO_DS)
DS_PROP = np.array([_ds_problem._property_name_from_optimization_pair(p) for p in _ds_problem.optimization_pairs])
P_DS = _ds_problem.n_params
X_DS = np.zeros((len(order), P_DS))
for k, e in enumerate(order):
    en = entries[e]
    X_DS[k] = h5["bounded_params"][int(en["param_offset"]) : int(en["param_offset"]) + P_DS]
h5.close()
print(f"dataset pool: {X_DS.shape[0]} starts from {TOPO_DS}", flush=True)

# ---------------- race problem ----------------
problem = UIFOProblem(size=3, topology=TOPO)
lower = np.array(problem.bounds)[0]
upper = np.array(problem.bounds)[1]
span = upper - lower
N = problem.n_params
PROP = np.array([problem._property_name_from_optimization_pair(p) for p in problem.optimization_pairs])

SIG = {f: float(X_DS[:, DS_PROP == f].std()) if (DS_PROP == f).sum() else 1.0 for f in FAMS}
print("family stds:", {k: round(v, 3) for k, v in SIG.items()}, flush=True)

POW_IDX_NP = np.where(PROP == "power")[0]
POW_IDX = jnp.array(POW_IDX_NP)
P_MAX = float(upper[POW_IDX_NP].max())  # raw power box cap (200)
print(f"power coords: {POW_IDX_NP.tolist()} (box cap {P_MAX})", flush=True)

def make_basins(K, seed):
    rng = np.random.default_rng(seed)
    B = np.zeros((K, N))
    n_ds = min(K // 2, X_DS.shape[0])
    for f in FAMS:
        mi = np.where(PROP == f)[0]
        di = np.where(DS_PROP == f)[0]
        if not len(mi) or not len(di):
            continue
        k = min(len(mi), len(di))
        for i in range(n_ds):
            src = X_DS[i]
            vals = src[di[:k]].copy()
            rng.shuffle(vals)
            B[i, mi[:k]] = vals
            if len(mi) > k:
                B[i, mi[k:]] = np.median(src[di])
    B[:n_ds] *= rng.normal(1.0, 0.05, (n_ds, N))
    B[:n_ds] += rng.normal(0.0, 0.01, (n_ds, N)) * span
    B[n_ds:] = lower + rng.random((K - n_ds, N)) * span
    return jnp.clip(jnp.array(B), jnp.array(lower), jnp.array(upper))

def scale_vec(mode):
    s = np.ones(N)
    if mode == "precond":
        for f in FAMS:
            m = PROP == f
            if m.sum():
                s[m] = max(SIG[f], 1e-3 * float(span[m].max()))
    return jnp.array(s)

S = scale_vec(CFG["transform"])
MID = jnp.array((lower + upper) / 2.0)
if CFG["transform"] == "precond":
    to_z = lambda x: (x - MID) / S
    to_x = lambda z: jnp.clip(z * S + MID, jnp.array(lower), jnp.array(upper))
else:
    to_z = lambda x: x - MID
    to_x = lambda z: jnp.clip(z + MID, jnp.array(lower), jnp.array(upper))

def OF(p):
    """Resolve the problem's aux objective AT CALL TIME — set_penalty_fn rebinds
    problem.objective_function_aux, and penalty fns bake into its trace."""
    return problem.objective_function_aux(p)

CHUNK = int(os.environ.get("L2D_CHUNK", "4"))  # GPU: chunk4 = 1.35 evals/s, OOM at 8 (Exp 07)

def run_chunks(fn, *args):
    outs = [fn(*(a[i : i + CHUNK] for a in args)) for i in range(0, args[0].shape[0], CHUNK)]
    return jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *outs)

SAFETY = float(CFG["safety"])
T_HARD, T_SOFT, T_DET = HARD_SIDE_POWER_THRESHOLD, SOFT_SIDE_POWER_THRESHOLD, DETECTOR_POWER_THRESHOLD

def c_vec(pv):
    """Log-space constraint values c_j = log(P_j/T_j): per sample -> (n_c,),
    batched (K, n_g, 1) -> (K, n_c). Group order: hard, detector, soft."""
    parts = [
        jnp.log((pv["hard"].squeeze(-1) + 1e-30) / T_HARD),
        jnp.log((pv["detector"].squeeze(-1) + 1e-30) / T_DET),
        jnp.log((pv["soft"].squeeze(-1) + 1e-30) / T_SOFT),
    ]
    return jnp.concatenate(parts, axis=-1)

def t_star_capped(pv, p):
    """Largest boundary-pinning joint power rescale (per-sample scalar, or batched
    (K,) when pv/p are batched): t* = safety*exp(-max_j c_j), capped so the power
    coords stay within [0, P_MAX]. >1 when the optics leave headroom — the
    projection spends it (S decreases in t); <1 restores feasibility."""
    c = c_vec(pv)
    m = jnp.max(c, axis=-1)
    t = SAFETY * jnp.exp(jnp.clip(-m, -30.0, 30.0))
    p_max = jnp.max(jnp.abs(p[..., POW_IDX]), axis=-1)
    return jnp.minimum(t, P_MAX / jnp.maximum(p_max, 1e-30))

def _proj_core(p):
    """Per-sample self-projected eval: evaluate at the boundary-scaled point.
    Returns (L2, (aux2, p2, t)); differentiable through the projection."""
    _, aux0 = OF(p)                       # fwd 1: powers at p
    t = t_star_capped(aux0["power_values"], p)
    p2 = p.at[POW_IDX].set(p[POW_IDX] * t)
    L2, aux2 = OF(p2)                     # fwd 2: loss + aux at the boundary point
    return L2, (aux2, p2, t)

def _alm_scalar(p, lam):
    """Per-sample augmented Lagrangian on log-space constraints c_j (<=0 feasible)."""
    _, aux0 = OF(p)
    c = c_vec(aux0["power_values"])
    rho = float(CFG["rho"])
    alm = jnp.sum((jax.nn.relu(lam + rho * c) ** 2 - lam**2) / (2.0 * rho))
    return aux0["sensitivity_loss"] + alm, aux0

def _head_scalar(p):
    """Per-sample softmax-max headroom objective (beta from config)."""
    _, aux0 = OF(p)
    c = c_vec(aux0["power_values"])
    beta = float(CFG["beta_softmax"])
    return jax.scipy.special.logsumexp(beta * c) / beta, aux0

proj_vg = jax.jit(jax.vmap(jax.value_and_grad(_proj_core, has_aux=True)))   # P1/P2/H1-ph2
proj_val = jax.jit(jax.vmap(_proj_core))                                    # PB logging stream: (L, (aux,p2,t))
vg0 = jax.jit(jax.vmap(jax.value_and_grad(lambda p: OF(p)[0])))             # raw-point grad (PB/D0/B1): (L, G)
vaux0 = jax.jit(jax.vmap(lambda p: OF(p)[1]))                               # raw-point aux (D0/B1/init)
alm_vg = jax.jit(jax.vmap(jax.value_and_grad(_alm_scalar, has_aux=True), in_axes=(0, 0)))
head_vg = jax.jit(jax.vmap(jax.value_and_grad(_head_scalar, has_aux=True)))

def _vag_split(out):
    """Unpack value_and_grad(f, has_aux=True) across jax layouts:
    jax 0.9: ((value, aux), grad); classic jax: (value, grad, aux)."""
    if len(out) == 3:
        v, g, a = out
    else:
        (v, a), g = out
    return v, g, a

BARRIER_MUS = [float(m) for m in CFG["barrier_mus"]]

def barrier_fn(mu):
    def fn(value, threshold):
        r = jnp.nan_to_num(value / threshold, nan=2.0, posinf=2.0, neginf=0.0)
        return jnp.where(r < 1.0, -mu * jnp.log1p(-jnp.clip(r, 0.0, 1.0 - 1e-12)), 1e3)
    return fn

PRESETS = {"squashed": squashed_relu_penalty, "relu": relu_penalty, "zero": zero_penalty}

def _ckpt(name, payload):
    # incremental checkpoint: survive any interruption
    with open(os.path.join(OUT, f"summary_{name}.json"), "w") as f:
        json.dump(payload, f, indent=2)

def _mask_power(updates):
    return updates.at[:, POW_IDX].set(0.0)

def run_arm(name, kind):
    """kind in: control | proj | proj_gauge | proj_logonly | barrier | alm | headroom"""
    K = CFG["K"]
    n_total = CFG["max_evals"] // K
    lr = CFG["lr"]
    noise0 = CFG["noise0"]
    restart_frac = CFG.get("warm_restart_every")

    B0 = make_basins(K, CFG["seed"])
    if kind == "barrier":
        # one-time boundary projection: start the barrier strictly interior
        aux0 = run_chunks(vaux0, B0)
        t0b = t_star_capped(aux0["power_values"], B0)
        B0 = B0.at[:, POW_IDX_NP].set(B0[:, POW_IDX_NP] * np.asarray(t0b)[:, None])
    Z = to_z(B0)

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    v_opt = jax.vmap(lambda g, st, p: optimizer.update(g, st, p))
    state = jax.vmap(optimizer.init)(Z)
    key = jax.random.PRNGKey(CFG["seed"])

    # phase plan: (phase_kind, steps, label, mask_power, kwargs)
    if kind == "control":
        phases = [("plain", int(round(fr * n_total)), f"{p}*{m}", False, dict(preset=p, mult=m))
                  for p, fr, m in CFG["D0_schedule"]]
    elif kind == "barrier":
        phases = [("plain", n_total // len(BARRIER_MUS), f"barrier_mu{mu}", False, dict(barrier=mu))
                  for mu in BARRIER_MUS]
    elif kind == "proj":
        phases = [("proj", n_total, "proj", False, {})]
    elif kind == "proj_gauge":
        phases = [("proj", n_total, "proj+mask", True, {})]
    elif kind == "proj_logonly":
        phases = [("projlog", n_total, "proj@log", False, {})]
    elif kind == "alm":
        phases = [("alm", n_total, "alm", False, {})]
    elif kind == "headroom":
        hf = float(CFG["headroom_frac"])
        phases = [("head", int(round(hf * n_total)), "headroom", True, {}),
                  ("proj", n_total - int(round(hf * n_total)), "proj", False, {})]
    else:
        raise ValueError(kind)
    if SMOKE:  # truncate every phase for fast scaffold validation
        phases = [(k_, min(s, 2), lab, msk, kw) for k_, s, lab, msk, kw in phases]

    # all non-control arms descend the pure sensitivity loss (zero penalty);
    # control sets its own preset(s) per phase below.
    problem.set_penalty_fn(zero_penalty)

    Lam = None
    n_feas = 0
    best = jnp.inf          # trajectory objective (arm-specific, diagnostic)
    best_feas = jnp.inf     # THE metric: min sensitivity loss over feasible evals
    hist = []
    tstars = []
    t0 = time.time()
    gstep = 0
    t_first_feas = None
    dual_every = int(CFG["dual_every"])
    box_frac = 0.0

    for ph_kind, steps, label, mask_power, kw in phases:
        if ph_kind == "plain":
            if "barrier" in kw:
                problem.set_penalty_fn(barrier_fn(kw["barrier"]))
            else:
                fn = PRESETS[kw["preset"]]
                mult = float(kw.get("mult", 1.0))
                problem.set_penalty_fn(fn if mult == 1.0 else
                                       (lambda v, t_, _f=fn, _m=mult: _m * _f(v, t_)))
            vg = jax.jit(jax.vmap(jax.value_and_grad(lambda p: OF(p)[0])))  # fresh trace per phase
            vaux = jax.jit(jax.vmap(lambda p: OF(p)[1]))
        if ph_kind == "alm":
            nc_probe = run_chunks(vaux0, to_x(Z)[:1])
            nc = int(c_vec(nc_probe["power_values"]).shape[-1])
            Lam = jnp.zeros((K, nc))

        obj = Objective(problem, max_time=1e9, max_evals=steps * K,
                        save=["batched_loss", "batched_is_feasible"])
        obj.start_logging()

        for it in range(steps):
            Xb = to_x(Z)
            if ph_kind == "proj":
                L, G, (aux, Xlog, tb) = _vag_split(run_chunks(proj_vg, Xb))
                Llog = aux["sensitivity_loss"]  # zero penalty => penalized loss
            elif ph_kind == "projlog":
                Lr, G = run_chunks(vg0, Xb)          # raw-point gradient (no constraint feedback)
                Lp, (aux, Xlog, tb) = run_chunks(proj_val, Xb)  # projected logging stream
                L = Lr                                # trajectory diagnostic
                Llog = aux["sensitivity_loss"]
            elif ph_kind == "alm":
                L, G, aux = _vag_split(run_chunks(alm_vg, Xb, Lam))
                Xlog, tb = Xb, None
                Llog = aux["sensitivity_loss"]
            elif ph_kind == "head":
                L, G, aux = _vag_split(run_chunks(head_vg, Xb))
                Xlog, tb = Xb, None
                Llog = aux["sensitivity_loss"]
            else:  # plain
                L, G = run_chunks(vg, Xb)
                aux = run_chunks(vaux, Xb)
                Xlog, tb = Xb, None
                Llog = L

            # diverged basins (resonant random inits): huge finite loss + frozen
            # grads so a single NaN basin cannot poison the batch or Adam state
            L = jnp.where(jnp.isfinite(L), L, 1e6)
            G = jnp.where(jnp.isfinite(G), G, 0.0)

            feas = aux["is_feasible"]
            sens = aux["sensitivity_loss"]
            bf = jnp.min(jnp.where(feas & jnp.isfinite(sens), sens, jnp.inf))
            best_feas = jnp.minimum(best_feas, bf)
            best = jnp.minimum(best, jnp.min(L))
            n_feas += int(jnp.sum(feas))
            obj.log_evaluation(params=Xlog, loss=Llog, aux=aux)

            if tb is None:  # host-side headroom diagnostic
                tb = t_star_capped(aux["power_values"], Xb)
            tstars.append(float(jnp.nanmean(tb)))
            if ph_kind in ("proj", "projlog"):
                box_frac = float(jnp.mean(jnp.asarray(Xlog)[:, POW_IDX_NP] > P_MAX - 1e-6))

            Gz = G / S
            if mask_power:
                Gz = _mask_power(Gz)
            updates, state = v_opt(Gz, state, Z)
            if noise0 > 0:
                nfrac = 1.0 - it / steps
                key, nk = jax.random.split(key)
                updates = updates + jax.random.normal(nk, Z.shape) * (noise0 * nfrac) * S / span
            if mask_power:
                updates = _mask_power(updates)
            Z = optax.apply_updates(Z, updates)

            if ph_kind == "alm" and (gstep + 1) % dual_every == 0:
                Lam = jax.nn.relu(Lam + float(CFG["rho"]) * c_vec(aux["power_values"]))

            if restart_frac and (it + 1) % max(1, int(restart_frac * steps)) == 0 and it + 1 < steps:
                k = K // 4
                order = jnp.argsort(L)
                Ztop = Z[order[:k]]
                key, nk1, nk2 = jax.random.split(key, 3)
                Z = Z.at[:k].set(Ztop)
                Z = Z.at[k:].set(Ztop[jax.random.randint(nk1, (K - k,), 0, k)] +
                                 jax.random.normal(nk2, (K - k, N)) * 0.05)
            gstep += 1
            hist.append((time.time() - t0, gstep, float(bf)))
            if t_first_feas is None and jnp.isfinite(bf):
                t_first_feas = time.time() - t0
            if gstep % 25 == 0:
                _ckpt(name, dict(name=name, kind=kind, K=K, iters=n_total,
                                 evals=CFG["max_evals"], wall_s=time.time() - t0,
                                 best=float(best), best_feasible=float(best_feas),
                                 n_feasible=n_feas, feas_frac=n_feas / (gstep * K),
                                 t_first_feas_s=t_first_feas, mean_tstar=tstars[-1],
                                 box_cap_frac=box_frac, phase=label, step=gstep,
                                 cfg=CFG, hist=hist, tstars=tstars))
        print(f"  [{name}] phase {label} done: best={float(best):.4f} "
              f"best_feas={float(best_feas):.4f} n_feas={n_feas} "
              f"mean_t*={tstars[-1]:.3g}", flush=True)

    return dict(name=name, kind=kind, K=K, iters=n_total, evals=CFG["max_evals"],
                wall_s=time.time() - t0, best=float(best), best_feasible=float(best_feas),
                n_feasible=n_feas, feas_frac=n_feas / (gstep * K), t_first_feas_s=t_first_feas,
                mean_tstar=tstars[-1] if tstars else None, box_cap_frac=box_frac,
                cfg=CFG, hist=hist, tstars=tstars)

ARMS = [
    dict(name="D0_control", kind="control"),
    dict(name="P1_proj_state", kind="proj"),
    dict(name="P2_gauge_fixed", kind="proj_gauge"),
    dict(name="PB_proj_logonly", kind="proj_logonly"),
    dict(name="B1_barrier", kind="barrier"),
    dict(name="L1_alm", kind="alm"),
    dict(name="H1_headroom", kind="headroom"),
]

results = []
# Optional arm selection for Slurm parallelism: L2D_ARMS="P1_proj_state,L1_alm" (unset = all).
_arm_filter = os.environ.get("L2D_ARMS")
if _arm_filter:
    _want = {s.strip() for s in _arm_filter.split(",")}
    ARMS = [a for a in ARMS if a["name"] in _want]
    print("arm filter:", sorted(_want), flush=True)

for a in ARMS:
    print(f"\n=== arm {a['name']} ({a['kind']}) ===", flush=True)
    r = run_arm(**a)
    r.pop("hist")
    r.pop("tstars")
    results.append(r)
    print(json.dumps(r, indent=1), flush=True)
    with open(os.path.join(OUT, f"summary_{a['name']}.json"), "w") as f:
        json.dump(r, f, indent=2)

if results:
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
print("\nsaved summary")
