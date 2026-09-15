"""Exp 07: K-scaling microbenchmark — batch-size ceiling on the allocated device.

Measures jitted vmap value_and_grad wall-time vs batch size K (GPU sbatch by
default; falls back to CPU if run on a compute node). The measured A5000 curve
gives the practical K ceiling; the H100 eval-env K is extrapolated via
TFLOP/bandwidth ratios with a safety margin (see AGENTS.md) and validated at
first H100 access. Single value_and_grad trace per K — matches what the race
arms actually run.

Env knobs:
  L2D_KS    comma list of K values (default 32,64,128,256,512,1024,2048,4096)
  L2D_REPS  timed reps per K (default 5)
  L2D_X64   1 = enable float64 (measure fp64 cost; default fp32)
Writes results/07_k_scaling/summary_<backend>.json
"""
import json
import os
import time

import numpy as np

import jax
import jax.numpy as jnp

from dfbench.problems import UIFOProblem

if os.environ.get("L2D_X64") == "1":
    jax.config.update("jax_enable_x64", True)

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results", "07_k_scaling")
os.makedirs(OUT, exist_ok=True)

KS = [int(k) for k in os.environ.get("L2D_KS", "32,64,128,256,512,1024,2048,4096").split(",")]
REPS = int(os.environ.get("L2D_REPS", "5"))
TOPO = "DBAHHHEGG-SLHLSSSLSLLL"
problem = UIFOProblem(size=3, topology=TOPO)
lower = np.array(problem.bounds)[0]
upper = np.array(problem.bounds)[1]
OF = problem.objective_function_aux
N = problem.n_params

dev = jax.devices()[0]
backend = jax.default_backend()
print(f"backend={backend} device={dev} x64={os.environ.get('L2D_X64') == '1'}", flush=True)

results = []
ceiling = None
for K in KS:
    Xb = jnp.array(lower + np.random.default_rng(K).random((K, N)) * (upper - lower))
    vg = jax.jit(jax.vmap(jax.value_and_grad(lambda p: OF(p)[0])))
    try:
        t0 = time.time()
        l, g = vg(Xb)
        jax.block_until_ready((l, g))
        t_compile = time.time() - t0
    except Exception as e:  # OOM ceiling: record and stop climbing
        print(f"K={K:5d}  FAILED: {type(e).__name__}: {e}", flush=True)
        results.append(dict(K=K, compile_s=None, per_step_s=None, evals_per_s=None,
                            error=type(e).__name__))
        ceiling = K
        break
    t0 = time.time()
    for _ in range(REPS):
        l, g = vg(Xb)
    jax.block_until_ready((l, g))
    dt = (time.time() - t0) / REPS
    results.append(dict(K=K, compile_s=t_compile, per_step_s=dt, evals_per_s=K / dt))
    print(f"K={K:5d}  compile {t_compile:7.1f}s  step {dt * 1e3:8.1f} ms  "
          f"{K / dt:9.0f} evals/s", flush=True)

with open(os.path.join(OUT, f"summary_{backend}.json"), "w") as f:
    json.dump(dict(backend=backend, device=str(dev), x64=os.environ.get("L2D_X64") == "1",
                   ks=KS, reps=REPS, ceiling_K=ceiling, results=results), f, indent=2)
print("saved")
