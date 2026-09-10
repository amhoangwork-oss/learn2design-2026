"""Exp 02+03 v2: Dataset validation + gradient conditioning.

Dataset scan showed 0/29650 entries match topology seed 42. So instead:
Part A (Exp 03): pick best entries across the dataset, rebuild each entry's OWN
  problem via UIFOProblem(size, topology=entry.topology_string), evaluate saved
  params, compare saved vs recomputed loss. Validates the dataset->problem->params
  pipeline (foundation for basin seeding / pretraining / conditioning stats).
Part B (Exp 02): per-property gradient conditioning at the best reproduced point,
  plus per-property value statistics over all 29,650 entries (metric R4).

Writes results/02_dataset_and_conditioning/{summary.json}
"""
import json
import os
import time

import h5py
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

from dfbench import Objective
from dfbench.problems import UIFOProblem

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results", "02_dataset_and_conditioning")
os.makedirs(OUT, exist_ok=True)

H5 = os.path.join(os.path.dirname(__file__), "..", "..", "Learn2Design-2026", "dataset", "dataset.h5")

h5 = h5py.File(H5, "r")
entries = h5["entries"][:]
losses_all = np.array(entries["loss"])
sizes = entries["size"][:]

scan = {
    "n_entries": int(len(entries)),
    "loss_all_min": float(losses_all.min()),
    "loss_all_p5": float(np.percentile(losses_all, 5)),
    "loss_all_median": float(np.median(losses_all)),
    "loss_all_max": float(losses_all.max()),
    "sizes": {int(s): int(c) for s, c in zip(*np.unique(sizes, return_counts=True))},
}

# per-property value statistics over ALL entries of size 3 (for metric R4).
# params are bounded; collect normalized position (p - lo)/(hi - lo) per property type.
# property identity per column needs a matching topology: only compare entries that
# share the most common topology -> gives per-property stats within a fixed layout.
topo = np.array([t.decode() if isinstance(t, bytes) else t for t in entries["topology_string"]])
utopos, counts = np.unique(topo, return_counts=True)
top_order = np.argsort(-counts)
scan["n_unique_topologies"] = int(len(utopos))
scan["top5_common_topologies"] = [
    {"topology": str(utopos[i]), "count": int(counts[i])} for i in top_order[:5]
]

# use the most common topology group for both param stats and reproduction check
TOPO = str(utopos[top_order[0]])
group_idx = np.where(topo == TOPO)[0]
group_losses = losses_all[group_idx]
scan["group_topology"] = TOPO
scan["group_n"] = int(len(group_idx))
scan["group_loss_min"] = float(group_losses.min())
scan["group_loss_median"] = float(np.median(group_losses))

# size of this group's entries
size_of_group = int(entries["size"][group_idx[0]])

# ---- param stats over the group (bounded + normalized) ----
P = int(entries["param_length"][group_idx[0]])
params_group = np.zeros((len(group_idx), P), dtype=np.float64)
for k, eidx in enumerate(group_idx):
    e = entries[eidx]
    off, ln = int(e["param_offset"]), int(e["param_length"])
    assert ln == P, (ln, P)
    params_group[k] = h5["bounded_params"][off : off + ln]

# ---- build problem, verify bounds alignment, evaluate best few ----
problem = UIFOProblem(size=size_of_group, topology=TOPO)
assert problem.n_params == P, (problem.n_params, P)
lower = np.array(problem.bounds)[0]
upper = np.array(problem.bounds)[1]

def prop_name(pair):
    return problem._property_name_from_optimization_pair(pair)

prop_names = np.array([prop_name(p_) for p_ in problem.optimization_pairs])

# per-property stats (physical units and normalized)
stats = {}
u = (params_group - lower) / (upper - lower)
for pname in sorted(set(prop_names)):
    m = prop_names == pname
    stats[pname] = dict(
        count=int(m.sum()),
        value_min=float(params_group[:, m].min()),
        value_median=float(np.median(params_group[:, m])),
        value_max=float(params_group[:, m].max()),
        value_std=float(params_group[:, m].std()),
        unorm_std=float(u[:, m].std()),
    )
scan["per_property_stats"] = stats

obj = Objective(problem, max_time=1800.0, save=["is_feasible", "sensitivity_loss", "penalty"])
obj.warmup_value_and_grad_aux()
obj.start_logging()

order = group_idx[np.argsort(losses_all[group_idx])]
take = order[: min(8, len(order))]
rows = []
for rank, eidx in enumerate(take):
    e = entries[eidx]
    off, ln = int(e["param_offset"]), int(e["param_length"])
    saved_loss = float(e["loss"])
    p = jnp.array(h5["bounded_params"][off : off + ln])
    t0 = time.time()
    loss, aux = obj.value_aux(p)
    dt = time.time() - t0
    rows.append(
        dict(
            rank=rank,
            entry_index=int(eidx),
            unique_hash=e["unique_hash"].decode() if isinstance(e["unique_hash"], bytes) else str(e["unique_hash"]),
            saved_loss=saved_loss,
            recomputed_loss=float(loss),
            sens=float(aux["sensitivity_loss"]),
            pen=float(aux["penalty"]),
            feasible=bool(aux["is_feasible"]),
            t_first_eval_s=dt,
        )
    )
    print(f"  entry {eidx}: saved={saved_loss:.4f} recomputed={float(loss):.4f} pen={float(aux['penalty']):.4f} feas={bool(aux['is_feasible'])} ({dt*1e3:.1f} ms)")

scan["max_abs_loss_diff"] = float(max(abs(r["saved_loss"] - r["recomputed_loss"]) for r in rows))
scan["eval_rows"] = rows

# ---- conditioning at best reproduced point ----
e0 = entries[take[0]]
p0 = jnp.array(h5["bounded_params"][int(e0["param_offset"]) : int(e0["param_offset"]) + int(e0["param_length"])])
loss0, grad0 = obj.value_and_grad(p0)
g = np.abs(np.array(grad0))
span = upper - lower
cond = {}
for pname in sorted(set(prop_names)):
    m = prop_names == pname
    cond[pname] = dict(
        grad_absmax=float(g[m].max()),
        grad_median=float(np.median(g[m])),
        grad_per_span_median=float(np.median(g[m] / span[m])),
    )
scan["conditioning_at_best"] = cond
scan["grad_norm_best"] = float(jnp.linalg.norm(grad0))

# normalized-coordinate gradient spread (R4 metric target)
scan["cond_ratio_absmax_over_median"] = float(g.max() / max(np.median(g), 1e-30))

with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(scan, f, indent=2)
print(json.dumps({k: v for k, v in scan.items() if k not in ("eval_rows",)}, indent=2)[:3000])
print("rows:", *rows, sep="\n  ")
