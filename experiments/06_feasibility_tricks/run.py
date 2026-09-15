"""Exp 06: feasibility tricks — zero-penalty descent vs squashed-penalty descent.

Question: on a hidden topology with unknown feasible-region location, does it pay
to run a first phase with zero_penalty (pure sensitivity descent, is_feasible
tracked separately), then flip to the default squashed penalty for repair?

Setup (same topology/pool machinery as Exp 05, K=128, 25600 evals each, CPU):
  A: squashed penalty all the way (competition default) — control
  B: zero penalty for first 40% of evals, then squashed
  C: zero penalty first 40%, then squashed * 4 (aggressive repair), last 10% squashed
Feasible count / best-feasible / time-to-first-feasible compared.

Writes results/06_feasibility_tricks/summary.json
"""
import json
import os
import time

import h5py
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import optax

from dfbench import Objective
from dfbench.problems import UIFOProblem
from dfbench.problems.base_problem import squashed_relu_penalty, zero_penalty

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results", "06_feasibility_tricks")
os.makedirs(OUT, exist_ok=True)

TOPO = "DBAHHHEGG-SLHLSSSLSLLL"
TOPO_DS = "DABFEEGBB-SLSLLSLSSDLS"
H5 = os.path.join(os.path.dirname(__file__), "..", "..", "Learn2Design-2026", "dataset", "dataset.h5")
FAMS = ["reflectivity", "tuning", "mass", "length", "power", "db", "angle"]

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

problem = UIFOProblem(size=3, topology=TOPO)
lower = np.array(problem.bounds)[0]
upper = np.array(problem.bounds)[1]
span = upper - lower
N = problem.n_params
PROP = np.array([problem._property_name_from_optimization_pair(p) for p in pairs]) if False else np.array(
    [problem._property_name_from_optimization_pair(p) for p in problem.optimization_pairs]
)

SIG = {f: float(X_DS[:, DS_PROP == f].std()) if (DS_PROP == f).sum() else 1.0 for f in FAMS}
s = np.ones(N)
for f in FAMS:
    m = PROP == f
    if m.sum():
        s[m] = max(SIG[f], 1e-3 * float(span[m].max()))
s = jnp.array(s)
mid = jnp.array((lower + upper) / 2.0)
to_z = lambda x: (x - mid) / s
to_x = lambda z: jnp.clip(z * s + mid, jnp.array(lower), jnp.array(upper))

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

OF = problem.objective_function_aux
CHUNK = 2  # CPU OOM probe: chunk2 traces ~10 min compile, 10.9 GB; chunk8 vg >25 min

def run_chunks(fn, Xb):
    outs = [fn(Xb[i : i + CHUNK]) for i in range(0, Xb.shape[0], CHUNK)]
    return jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *outs)

_vg = jax.jit(jax.vmap(jax.value_and_grad(lambda p: OF(p)[0])))
_vaux = jax.jit(jax.vmap(lambda p: OF(p)[1]))

def vg_batch(Xb):
    return run_chunks(_vg, Xb)

def aux_batch(Xb):
    return run_chunks(_vaux, Xb)

def run_arm(name, zero_frac, aggressive=1.0, K=128, seed=7, lr=0.05, noise0=0.3, max_evals=25600):
    obj = Objective(problem, max_time=3600.0, max_evals=max_evals,
                    save=["batched_loss", "batched_is_feasible"])
    # penalty schedule: zero for zero_frac, then squashed (possibly scaled)
    # implement via set_penalty_fn: zero_penalty for phase 1; after that switch.
    # set_penalty_fn re-traces: do it ONCE mid-run (allowed: only before logging? -> NO,
    # it raises after logging). So: build TWO problems? Same problem object is shared.
    # Solution: run phase 1 with zero_penalty set BEFORE start_logging, then rebuild a
    # second Objective sharing the problem AFTER set_penalty_fn(squashed) and start
    # logging of a fresh Objective for phase 2. Budget continuity handled by max_evals.
    n_steps = max_evals // K
    z_steps1 = int(zero_frac * n_steps)

    # --- phase 1: zero penalty ---
    problem.set_penalty_fn(zero_penalty)
    obj1 = Objective(problem, max_time=3600.0, max_evals=int(zero_frac * max_evals),
                     save=["batched_loss", "batched_is_feasible"])
    obj1.start_logging()

    B0 = make_basins(K, seed)
    Z = to_z(B0)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    v_opt = jax.vmap(lambda g, st, p: optimizer.update(g, st, p))
    state = jax.vmap(optimizer.init)(Z)
    key = jax.random.PRNGKey(seed)

    n_feas = 0
    best_feas = jnp.inf
    best = jnp.inf
    hist = []
    t0 = time.time()
    for it in range(z_steps1):
        Xb = to_x(Z)
        L, G = vg_batch(Xb)
        aux = aux_batch(Xb)
        feas = aux["is_feasible"]
        best = jnp.minimum(best, jnp.min(L))
        bf = jnp.min(jnp.where(feas, L, jnp.inf))
        best_feas = jnp.minimum(best_feas, bf)
        n_feas += int(jnp.sum(feas))
        obj1.log_evaluation(params=Xb, loss=L, aux=aux)
        updates, state = v_opt(G / s, state, Z)
        frac = 1.0 - it / z_steps1
        key, nk = jax.random.split(key)
        updates = updates + jax.random.normal(nk, Z.shape) * (noise0 * frac) * 0.01
        Z = optax.apply_updates(Z, updates)
        hist.append((time.time() - t0, float(bf)))

    # --- phase 2: squashed penalty (fresh Objective for clean logging) ---
    problem.set_penalty_fn(squashed_relu_penalty)
    obj2 = Objective(problem, max_time=3600.0, max_evals=max_evals - int(zero_frac * max_evals),
                     save=["batched_loss", "batched_is_feasible"])
    obj2.start_logging()
    for it in range(n_steps - z_steps1):
        Xb = to_x(Z)
        L, G = vg_batch(Xb)
        aux = aux_batch(Xb)
        feas = aux["is_feasible"]
        bf = jnp.min(jnp.where(feas, L, jnp.inf))
        best_feas = jnp.minimum(best_feas, bf)
        n_feas += int(jnp.sum(feas))
        obj2.log_evaluation(params=Xb, loss=L, aux=aux)
        updates, state = v_opt(G / s, state, Z)
        if it < 0.7 * (n_steps - z_steps1):
            frac = 1.0 - it / (0.7 * (n_steps - z_steps1))
            key, nk = jax.random.split(key)
            updates = updates + jax.random.normal(nk, Z.shape) * (noise0 * frac) * 0.01
        Z = optax.apply_updates(Z, updates)
        hist.append((time.time() - t0, float(bf)))

    wall = time.time() - t0
    return dict(name=name, zero_frac=zero_frac, aggressive=aggressive, K=K,
                iters=n_steps, evals=max_evals, wall_s=wall,
                best=float(best), best_feasible=float(best_feas), n_feasible=n_feas,
                feas_frac=n_feas / max_evals, hist=hist)

ARMS = [
    dict(name="A_squashed_always", zero_frac=0.0, K=64, max_evals=12800),
    dict(name="B_zero40_then_squashed", zero_frac=0.4, K=64, max_evals=12800),
]

results = []
# Optional arm selection for Slurm parallelism: L2D_ARMS="A_squashed_always" (comma-separated; unset = all).
_arm_filter = os.environ.get("L2D_ARMS")
if _arm_filter:
    _want = {s.strip() for s in _arm_filter.split(",")}
    ARMS = [a for a in ARMS if a["name"] in _want]
    print("arm filter:", sorted(_want), flush=True)

for a in ARMS:
    print(f"\n=== arm {a['name']} ===", flush=True)
    r = run_arm(**a)
    r.pop("hist")
    results.append(r)
    print(json.dumps(r, indent=1), flush=True)
    with open(os.path.join(OUT, f"summary_{a['name']}.json"), "w") as f:
        json.dump(r, f, indent=2)

if results:
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
print("\nsaved")
