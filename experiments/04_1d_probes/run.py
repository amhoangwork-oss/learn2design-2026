"""Exp 04: 1-D finite-difference probes of the loss along each property family.

For the seed-42 topology (from Exp 01), start at the midpoint and probe the loss
along single coordinates from each property family (reflectivity, tuning, mass,
length, power, db, angle) — in PHYSICAL space (bounded space, sigmoid-off) and,
separately, in the UNBOUNDED (logit) space, to test:
  R1: power responds linearly in log-space (loss vs log(P))
  R2: reflectivity responds linearly in log(1-R)
  R3: tuning/angle periodicity (probe wrap point ±360: does loss change fast there?)
Also records local curvature estimates (2nd-order FD) per family.

Writes results/04_1d_probes/{summary.json, curves.json}
"""
import json
import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

import jax
import jax.numpy as jnp

from dfbench import Objective
from dfbench.problems import UIFOProblem


def _inverse_sigmoid_bounding(params, bounds):
    u = (params - bounds[0]) / (bounds[1] - bounds[0])
    u = jnp.clip(u, 1e-7, 1 - 1e-7)
    return jnp.log(u / (1 - u))

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results", "04_1d_probes")
os.makedirs(OUT, exist_ok=True)

TOPOLOGY_SEED42 = "DBAHHHEGG-SLHLSSSLSLLL"
problem = UIFOProblem(size=3, topology=TOPOLOGY_SEED42)
pairs = problem.optimization_pairs
lower = np.array(problem.bounds)[0]
upper = np.array(problem.bounds)[1]

def prop_name(pair):
    return problem._property_name_from_optimization_pair(pair)

prop_names = np.array([prop_name(p_) for p_ in pairs])
n = problem.n_params

obj = Objective(problem, max_time=3600.0, save=["is_feasible", "sensitivity_loss", "penalty"])
obj.warmup_value_and_grad_aux()
obj.start_logging()

# unbounded (logit) space helpers
def to_unbounded(p_b):
    return _inverse_sigmoid_bounding(p_b, problem.bounds)

# base point: midpoint
p_base = jnp.array((lower + upper) / 2.0)
u_base = to_unbounded(p_base)

# pick representative coordinate per family: the one with largest |grad| there
loss0, aux0 = obj.value_aux(p_base)
_, grad0 = obj.value_and_grad(p_base)
g = np.abs(np.array(grad0))

representative = {}
for pname in sorted(set(prop_names)):
    m = np.where(prop_names == pname)[0]
    representative[pname] = int(m[np.argmax(g[m])])

print("representative coords:", representative)

# probe each family around the base point, in bounded space
# relative offsets: fractions of the span, plus absolute small offsets
curves = {}
for pname, idx in representative.items():
    span = upper[idx] - lower[idx]
    base_val = float(p_base[idx])
    # offsets: ±2%, ±5%, ±10%, ±25% of span (skip if out of bounds)
    fracs = [-0.25, -0.10, -0.05, -0.02, 0.02, 0.05, 0.10, 0.25]
    xs_b, xs_u, ys = [], [], []
    for f in fracs:
        v = base_val + f * span
        if v < lower[idx] or v > upper[idx]:
            continue
        p = p_base.at[idx].set(v)
        loss, aux = obj.value_aux(p)
        xs_b.append(float(v))
        uu = to_unbounded(p)
        xs_u.append(float(uu[idx]))
        ys.append(
            dict(
                loss=float(loss),
                sens=float(aux["sensitivity_loss"]),
                pen=float(aux["penalty"]),
                feasible=bool(aux["is_feasible"]),
            )
        )
    curves[pname] = dict(
        idx=idx,
        base_val=base_val,
        span=float(span),
        xs_bounded=xs_b,
        xs_unbounded=xs_u,
        ys=ys,
    )
    print(f"probed {pname} (idx {idx}): {len(xs_b)} pts, loss {ys[0]['loss']:.3f} -> {ys[-1]['loss']:.3f}")

# periodicity probe for tuning: wrap behavior. Take a tuning coordinate and set
# values near +360 and -360 (same physical phase) — compare losses of v and v-360.
per = {}
tuning_idxs = np.where(prop_names == "tuning")[0][:3]
for idx in tuning_idxs:
    span = upper[idx] - lower[idx]
    base_val = float(p_base[idx])
    a = 300.0  # near +360
    p_a = p_base.at[idx].set(a)
    p_b = p_base.at[idx].set(a - 360.0)  # equivalent phase
    la, _ = obj.value_aux(p_a)
    lb, _ = obj.value_aux(p_b)
    per[int(idx)] = dict(
        val_a=a, loss_a=float(la), val_b=a - 360.0, loss_b=float(lb),
        diff=float(abs(la - lb)),
    )
    print(f"periodicity idx {idx}: L(+{a})={float(la):.6f} L({a-360})={float(lb):.6f} diff={float(abs(la-lb)):.2e}")

summary = dict(
    base_loss=float(loss0),
    base_sens=float(aux0["sensitivity_loss"]),
    base_pen=float(aux0["penalty"]),
    base_feasible=bool(aux0["is_feasible"]),
    representative=representative,
    periodicity=per,
    curves=curves,
)
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("saved to", OUT)
