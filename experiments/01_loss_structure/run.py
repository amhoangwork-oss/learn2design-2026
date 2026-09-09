"""Exp 01: Instantiate UIFOProblem on CPU; inspect params/bounds; first loss+grad; timing.

Writes results/01_loss_structure/summary.json
"""
import json
import time
import os

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

from dfbench import Objective
from dfbench.problems import UIFOProblem

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "01_loss_structure")
os.makedirs(OUT, exist_ok=True)

t0 = time.time()
problem = UIFOProblem(size=3, topology_seed=42)
t_init = time.time() - t0

n = problem.n_params
bounds = np.array(problem.bounds)
pairs = problem.optimization_pairs

# Property distribution
from collections import Counter

prop_names = [problem._property_name_from_optimization_pair(p) for p in pairs]
prop_counts = Counter(prop_names)

# Coupled parameters (shared across components)
n_coupled = sum(1 for p in pairs if isinstance(p[0], list))

obj = Objective(problem, max_time=600.0, save=["is_feasible"])

# Midpoint params for smoke evaluation
mid = jnp.array((bounds[0] + bounds[1]) / 2.0)

obj.warmup_value_and_grad()  # JIT compile before logging (not budgeted)
obj.start_logging()

t0 = time.time()
loss = obj.value(mid)
t_first = time.time() - t0
t0 = time.time()
loss2 = obj.value(mid)
t_second = time.time() - t0

t0 = time.time()
loss_g, grad = obj.value_and_grad(mid)
t_vg_first = time.time() - t0
t0 = time.time()
loss_g2, grad2 = obj.value_and_grad(mid)
t_vg_second = time.time() - t0

# aux at midpoint
loss_a, aux = obj.value_aux(mid)

summary = {
    "topology_seed": 42,
    "topology_string": problem.topology_string,
    "size": 3,
    "homodyne": problem._homodyne,
    "n_params": int(n),
    "n_coupled": int(n_coupled),
    "prop_counts": dict(prop_counts),
    "bounds_min": bounds[0].tolist(),
    "bounds_max": bounds[1].tolist(),
    "unique_bounds": sorted(set(map(tuple, bounds.T.tolist()))),
    "loss_midpoint": float(loss),
    "loss_vg": float(loss_g),
    "aux_sensitivity_loss": float(aux["sensitivity_loss"]),
    "aux_penalty": float(aux["penalty"]),
    "aux_is_feasible": bool(aux["is_feasible"]),
    "t_init_s": t_init,
    "t_first_eval_s": t_first,
    "t_second_eval_s": t_second,
    "t_first_valuegrad_s": t_vg_first,
    "t_second_valuegrad_s": t_vg_second,
    "grad_norm": float(jnp.linalg.norm(grad)),
    "grad_absmax": float(jnp.max(jnp.abs(grad))),
    "jax_backend": str(jax.devices()),
}
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
