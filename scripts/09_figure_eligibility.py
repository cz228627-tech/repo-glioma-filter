"""Figure 1: patient x state eligibility and its relation to cell recovery (explicit drawing)."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

M = pd.read_csv(os.path.join(C.OUT, "states10_matrix.csv"), index_col=0)
pairs = pd.read_csv(os.path.join(C.OUT, "care_wt_pairs.csv")).set_index("patient")
cells = pairs.loc[M.index, ["tp_total", "r1_total"]].min(axis=1)
n = M.sum(axis=1)

order = cells.sort_values().index
Mo = M.loc[order].astype(int)
co = cells.loc[order]

fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.6), gridspec_kw={"width_ratios": [1.45, 1]})
a = ax[0]
nx, ny = Mo.shape[1], Mo.shape[0]
a.set_xlim(0, nx)
a.set_ylim(ny, 0)
a.set_facecolor("white")

for i in range(ny):
    for j in range(nx):
        if Mo.iloc[i, j] == 1:
            a.add_patch(Rectangle((j, i), 1, 1, facecolor="0.15", edgecolor="none"))

for j in range(nx + 1):
    a.plot([j, j], [0, ny], color="0.85", lw=0.4, zorder=3)
for i in range(ny + 1):
    a.plot([0, nx], [i, i], color="0.85", lw=0.4, zorder=3)

a.add_patch(Rectangle((0, 0), nx, ny, fill=False, edgecolor="0.3", lw=0.8, zorder=4))
zero = int((n == 0).sum())
a.axhline(zero, color="tab:red", lw=0.9, ls="--", zorder=5)
a.text(nx - 0.15, zero - 0.6, f"{zero} patients with no\neligible state", fontsize=6.5,
       color="tab:red", ha="right", va="bottom")
a.set_xticks(np.arange(nx) + 0.5)
a.set_xticklabels(Mo.columns, rotation=45, ha="right", fontsize=8)
a.set_yticks([])
a.set_ylabel("Patients (sorted by malignant cells recovered)", fontsize=8)
a.set_title("a  Eligibility by state (20 cells at both endpoints; filled = eligible)",
            fontsize=8, loc="left")

b = ax[1]
b.scatter(cells, n, s=14, color="k", alpha=0.55, linewidth=0)
b.set_xscale("log")
b.set_xlabel("Malignant cells per pair (lower endpoint)", fontsize=8)
b.set_ylabel("States eligible per patient", fontsize=8)
rho = spearmanr(cells, n)
b.text(0.03, 0.95, f"\u03c1 = {rho.statistic:.2f}\nP = {rho.pvalue:.1e}", transform=b.transAxes,
       fontsize=8, va="top")
b.set_title("b  Eligibility clusters within patients", fontsize=8, loc="left")

for x in ax:
    x.tick_params(labelsize=7)
    for s in x.spines.values():
        s.set_linewidth(0.6)

fig.tight_layout()
out = os.path.join(C.OUT, "figures", "FigEligibility.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=400, bbox_inches="tight")
print("wrote", out, "| eligible cells drawn:", int(Mo.to_numpy().sum()), "of", nx * ny)
