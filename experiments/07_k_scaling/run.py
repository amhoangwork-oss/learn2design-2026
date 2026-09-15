"""Exp 07: chunked-vmap throughput sweep on the allocated GPU.

First GPU attempt (2026-09-15) established: the UIFO sim computes in float64
natively (f64[.,50,~700,~700] propagation tensors, ~200 MB/sample forward)
regardless of jax_enable_x64, and an UNCHUNKED vmap value_and_grad OOMs on a
24 GB A5000 at K=32 (6.4 GB single intermediate + autotuner workspace). The
practical knob is therefore the vmap CHUNK size — Exp 05/06 loop chunks
(run_chunks) to reach the total basin count K, so K itself is unbounded.

Sweep: compile time, step time, throughput, peak memory per chunk size; an OOM
records the ceiling. H100 eval-env transfer is bandwidth-bound (f64 tensors):
A5000 768 GB/s -> H100 2039 (PCIe) / 3352 (SXM) GB/s = 2.65x / 4.36x nominal;
with a 0.5x safety margin the A5000-optimal chunk is the safe floor, ~1.3-2.2x
larger likely. (FP64 compute ratio 60-78x is the upper bound if compute-bound.)

Env knobs:
  L2D_CHUNKS  comma list of chunk sizes (default 1,2,4,8,16,32)
  L2D_REPS    timed reps per chunk (default 5)
Writes results/07_k_scaling/summary_gpu.json
"""
import json
import os
import time

import numpy as np

import jax
import jax.numpy as jnp

from dfbench.problems import UIFOProblem

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results", "07_k_scaling")
os.makedirs(OUT, exist_ok=True)

CHUNKS = [int(c) for c in os.environ.get("L2D_CHUNKS", "1,2,4,8,16,32").split(",")]
REPS = int(os.environ.get("L2D_REPS", "5"))
TOPO = "DBAHHHEGG-SLHLSSSLSLLL"
problem = UIFOProblem(size=3, topology=TOPO)
lower = np.array(problem.bounds)[0]
upper = np.array(problem.bounds)[1]
OF = problem.objective_function_aux
N = problem.n_params

dev = jax.devices()[0]
backend = jax.default_backend()
print(f"backend={backend} device={dev}", flush=True)

results = []
best_chunk = None
for C in CHUNKS:
    Xb = jnp.array(lower + np.random.default_rng(C).random((C, N)) * (upper - lower))
    vg = jax.jit(jax.vmap(jax.value_and_grad(lambda p: OF(p)[0])))
    try:
        t0 = time.time()
        l, g = vg(Xb)
        jax.block_until_ready((l, g))
        t_compile = time.time() - t0
    except Exception as e:  # OOM ceiling: record and stop climbing
        print(f"CHUNK={C:4d}  FAILED: {type(e).__name__}", flush=True)
        results.append(dict(chunk=C, compile_s=None, per_step_s=None,
                            evals_per_s=None, peak_mem_gb=None,
                            error=type(e).__name__))
        break
    t0 = time.time()
    for _ in range(REPS):
        l, g = vg(Xb)
    jax.block_until_ready((l, g))
    dt = (time.time() - t0) / REPS
    mem = dev.memory_stats() if hasattr(dev, "memory_stats") else {}
    peak_gb = mem.get("peak_bytes_in_use", 0) / 2**30
    results.append(dict(chunk=C, compile_s=t_compile, per_step_s=dt,
                        evals_per_s=C / dt, peak_mem_gb=peak_gb, error=None))
    print(f"CHUNK={C:4d}  compile {t_compile:7.1f}s  step {dt * 1e3:8.1f} ms  "
          f"{C / dt:9.0f} evals/s  peak {peak_gb:5.1f} GB", flush=True)
    best_chunk = C

with open(os.path.join(OUT, f"summary_{backend}.json"), "w") as f:
    json.dump(dict(backend=backend, device=str(dev), chunks=CHUNKS, reps=REPS,
                   best_chunk=best_chunk, results=results), f, indent=2)
print("saved")
