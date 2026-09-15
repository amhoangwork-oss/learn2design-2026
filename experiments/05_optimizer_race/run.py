"""Exp 05: optimizer horse-race on UIFO seed 42 (CPU).

Arms (187 params; one fixed vmap trace per arm; identical budgets):
  C0: Adam  raw,             K=64   noise off
  C1: Adam  raw+noise,       K=64   noise 0.3->0 (z-space, s=span)
  P0: Adam  preconditioned,  K=64   noise off
  P1: Adam  precond+noise,   K=64   noise 0.3->0
  PB: Adam  precond+noise,   K=256  (batch-size arm)

Basins: half dataset warm starts (family-matched from the 2878-entry
DABFEEGBB-SLSLLSLSSDLS pool), half random; 5% multiplicative jitter.
Per-basin independent Adam state via vmap-of-updates. Loss: default squashed
penalty (schedule comparison lives in Exp 06). Feasibility from aux.

Writes results/05_optimizer_race/{summary.json, history.csv}
"""
import csv
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

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results", "05_optimizer_race")
os.makedirs(OUT, exist_ok=True)

TOPO = "DBAHHHEGG-SLHLSSSLSLLL"
TOPO_DS = "DABFEEGBB-SLSLLSLSSDLS"
H5 = os.path.join(os.path.dirname(__file__), "..", "..", "Learn2Design-2026", "dataset", "dataset.h5")
FAMS = ["reflectivity", "tuning", "mass", "length", "power", "db", "angle"]

# ---------------- dataset warm-start pool ----------------
h5 = h5py.File(H5, "r")
entries = h5["entries"][:]
topo = np.array([t.decode() if isinstance(t, bytes) else t for t in entries["topology_string"]])
idx = np.where(topo == TOPO_DS)[0]
losses_ds = np.array(entries["loss"])[idx]
order = idx[np.argsort(losses_ds)]

_ds_problem = UIFOProblem(size=3, topology=TOPO_DS)
DS_PROP = np.array([_ds_problem._property_name_from_optimization_pair(p) for p in _ds_problem.optimization_pairs])
P_DS = _ds_problem.n_params
X_DS = np.zeros((len(order), P_DS))
for k, e in enumerate(order):
    en = entries[e]
    X_DS[k] = h5["bounded_params"][int(en["param_offset"]) : int(en["param_offset"]) + P_DS]
h5.close()
print(f"dataset pool: {X_DS.shape[0]} starts from {TOPO_DS}")

# ---------------- race problem ----------------
problem = UIFOProblem(size=3, topology=TOPO)
pairs = problem.optimization_pairs
lower = np.array(problem.bounds)[0]
upper = np.array(problem.bounds)[1]
span = upper - lower
N = problem.n_params
PROP = np.array([problem._property_name_from_optimization_pair(p) for p in pairs])

SIG = {f: float(X_DS[:, DS_PROP == f].std()) if (DS_PROP == f).sum() else 1.0 for f in FAMS}
print("family stds:", {k: round(v, 3) for k, v in SIG.items()})

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

OF = problem.objective_function_aux
# Chunked vmap: XLA fuses the vmapped linear solves into giant buffers (OOM at
# K>=8 unchunked on 39GB RAM). CHUNK=2 traces in ~10 min at 10.9 GB (probe:
# chunk8 vg compile exceeded 25 min). All batch calls go through run_chunks.
CHUNK = int(os.environ.get("L2D_CHUNK", "2"))  # CPU OOM probe: chunk2 traces ~10 min compile, 10.9 GB; chunk8 vg >25 min

def run_chunks(fn, Xb):
    outs = [fn(Xb[i : i + CHUNK]) for i in range(0, Xb.shape[0], CHUNK)]
    return jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *outs)

_vg = jax.jit(jax.vmap(jax.value_and_grad(lambda p: OF(p)[0])))
_vaux = jax.jit(jax.vmap(lambda p: OF(p)[1]))

def vg_batch(Xb):
    return run_chunks(_vg, Xb)

def aux_batch(Xb):
    return run_chunks(_vaux, Xb)

def run_arm(name, mode, K, seed, lr, noise0, max_evals, warm_restart_every=None):
    obj = Objective(problem, max_time=3600.0, max_evals=max_evals,
                    save=["batched_loss", "batched_is_feasible"])
    obj.start_logging()

    s = scale_vec(mode)
    mid = jnp.array((lower + upper) / 2.0)
    if mode == "precond":
        to_z = lambda x: (x - mid) / s
        to_x = lambda z: jnp.clip(z * s + mid, jnp.array(lower), jnp.array(upper))
    else:
        to_z = lambda x: x - mid
        to_x = lambda z: jnp.clip(z + mid, jnp.array(lower), jnp.array(upper))

    B0 = make_basins(K, seed)
    Z = to_z(B0)

    # per-basin independent Adam state
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    v_opt = jax.vmap(lambda g, st, p: optimizer.update(g, st, p))

    # init state per basin: optax states are pytrees; use init on a batch via vmap
    init_state = jax.vmap(optimizer.init)(Z)

    key = jax.random.PRNGKey(1000 + seed)
    n_steps = max_evals // K
    noise_steps = max(1, int(0.7 * n_steps))
    restart_every = max(1, int(warm_restart_every * n_steps)) if warm_restart_every else 0

    best = jnp.inf
    best_feas = jnp.inf
    n_feas = 0
    hist = []
    t0 = time.time()
    first_feas_seen = False
    t_first_feas = None

    for it in range(n_steps):
        Xb = to_x(Z)
        (L, G) = vg_batch(Xb)
        aux = aux_batch(Xb)
        feas = aux["is_feasible"]

        best = jnp.minimum(best, jnp.min(L))
        bf = jnp.min(jnp.where(feas, L, jnp.inf))
        best_feas = jnp.minimum(best_feas, bf)
        n_feas += int(jnp.sum(feas))

        obj.log_evaluation(params=Xb, loss=L, aux=aux)

        # per-basin update
        Gz = G / s  # chain rule into z-space
        updates, init_state = v_opt(Gz, init_state, Z)
        if noise0 > 0 and it < noise_steps:
            frac = 1.0 - it / noise_steps
            key, nk = jax.random.split(key)
            updates = updates + jax.random.normal(nk, Z.shape) * (noise0 * frac) * s / span
        Z = optax.apply_updates(Z, updates)

        if restart_every and (it + 1) % restart_every == 0 and it + 1 < n_steps:
            # keep top quartile, re-perturb the rest around them
            k = K // 4
            order = jnp.argsort(L)
            Ztop = Z[order[:k]]
            key, nk1, nk2 = jax.random.split(key, 3)
            Z = Z.at[:k].set(Ztop)
            Z = Z.at[k:].set(Ztop[jax.random.randint(nk1, (K - k,), 0, k)] +
                             jax.random.normal(nk2, (K - k, N)) * 0.05 * s / span * span)

        wb = float(bf)
        hist.append((time.time() - t0, it, wb))
        if not first_feas_seen and jnp.isfinite(bf):
            first_feas_seen = True
            t_first_feas = time.time() - t0

    wall = time.time() - t0
    obj.finalize_display()
    return dict(
        name=name, mode=mode, K=K, iters=n_steps, evals=max_evals, wall_s=wall,
        best=float(best), best_feasible=float(best_feas), n_feasible=n_feas,
        feas_frac=n_feas / max_evals, t_first_feas_s=t_first_feas,
        hist=hist,
    )

ARMS = [
    dict(name="C0_raw_adam_K64",        mode="raw",      K=64,  seed=1, lr=0.05, noise0=0.0, max_evals=12800),
    dict(name="C1_raw_adam_noise_K64",  mode="raw",      K=64,  seed=2, lr=0.05, noise0=0.3, max_evals=12800),
    dict(name="P0_prec_adam_K64",       mode="precond",  K=64,  seed=3, lr=0.05, noise0=0.0, max_evals=12800),
    dict(name="P1_prec_noise_K64",      mode="precond",  K=64,  seed=4, lr=0.05, noise0=0.3, max_evals=12800,
         warm_restart_every=0.25),
    dict(name="PB_prec_noise_K128",     mode="precond",  K=128, seed=5, lr=0.05, noise0=0.3, max_evals=12800,
         warm_restart_every=0.25),
]

results = []
# Optional arm selection for Slurm parallelism: L2D_ARMS="P0_prec_adam_K64,PB_prec_noise_K128"
# (comma-separated names; unset = run all arms in one process).
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
print("\nsaved summary")
