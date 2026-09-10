"""Plot 1-D probe curves from Exp 04 into a single figure."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
s = json.load(open(os.path.join(HERE, "..", "results", "04_1d_probes", "summary.json")))

families = ["reflectivity", "tuning", "power", "length", "mass", "db", "angle"]
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.ravel()

for k, pname in enumerate(families):
    ax = axes[k]
    c = s["curves"][pname]
    xs = np.array(c["xs_bounded"])
    ys = np.array([y["loss"] for y in c["ys"]])
    o = np.argsort(xs)
    base = c["base_val"]
    # x-axis: relative offset from base in % of span
    span = c["span"]
    rel = 100 * (xs[o] - base) / span
    ax.plot(rel, ys[o], "o-", lw=2, ms=5, color="#3b6fd4")
    ax.axvline(0, color="gray", ls=":", lw=1)
    ax.set_title(f"{pname} (idx {c['idx']}, base {base:.3g})", fontsize=11)
    ax.set_xlabel("offset [% of bound span]")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.3)

axes[7].axis("off")
txt = (
    "Base point: midpoint of topology seed42\n"
    f"base loss {s['base_loss']:.3f} = sens {s['base_sens']:.2f} + pen {s['base_pen']:.2f}\n\n"
    "Key: reflectivity has a sharp valley\n"
    "(Δ0.89 loss across ±25% span);\n"
    "power ~linear in log P (0.65/decade);\n"
    "tuning phase-structured at 90°;\n"
    "mass/db/angle locally flat."
)
axes[7].text(0.02, 0.72, txt, fontsize=11, va="top")

fig.suptitle("Learn2Design 1-D loss probes per property family (UIFO seed 42, CPU)", fontsize=13)
fig.tight_layout()
out = os.path.join(HERE, "..", "results", "04_1d_probes", "probe_curves.png")
fig.savefig(out, dpi=110)
print("saved", out)
