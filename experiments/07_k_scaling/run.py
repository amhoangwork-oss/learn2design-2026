"""Exp 07: K-scaling microbenchmark (batch-size feasibility on real hardware).

Measures vmap_value_and_grad_aux wall-time vs batch size on CPU to find the
practical K ceiling. Same measure rerun later on GPU (H100 eval env) gives the
competition number. Fixed trace: one compile per batch size, T reps each.

Writes results/07_k_scaling/{summary.json}
"""
import json
import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

import jax
import jax.numpy as jnp

from dfbench.problems import UIFOProblem

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results", "07_k_scaling")
os.makedirs(OUT, exist_ok=True)

TOPO = "DBAHHHEGG-SLHLSSSLSLLL"
problem = UIFOProblem(size=3, topology=TOPO)
lower = np.array(problem.bounds)[0]
upper = np.array(problem.bounds)[1]
OF = problem.objective_function_aux

results = []
for K in [32, 64, 128, 256, 512, 1024, 2048]:
    Xb = jnp.array(lower + np.random.default_rng(K).random((K, problem.n_params)) * (upper - lower))
    f = jax.jit(jax.vmap(lambda p: OF(p)[0]))
    g = jax.jit(jax.vmap(jax.grad(lambda p: OF(p)[0])))
    t0 = time.time()
    _ = f(Xb); _ = g(Xb)
    jax.block_until_ready(_)
    t_compile = time.time() - t0
    # timed reps (5x value+grad)
    t0 = time.time()
    reps = 5
    for _ in range(reps):
        l = f(Xb); gg = g(Xb)
    jax.block_until_ready(l); jax.block_until_ready(gg)
    dt = (time.time() - t0) / reps
    results.append(dict(K=K, compile_s=t_compile, per_step_s=dt, evals_per_s=K / dt))
    print(f"K={K:5d}  compile {t_compile:7.1f}s  step {dt*1e3:8.1f} ms  {K/dt:9.0f} evals/s", flush=True)

with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(results, f, indent=2)
print("saved")
