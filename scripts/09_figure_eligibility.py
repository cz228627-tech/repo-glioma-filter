"""Figure: patient x state eligibility and its relation to cell recovery."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

M = pd.read_csv(os.path.join(C.OUT, "states10_matrix.csv"), index_col=0)
pairs = pd.read_csv(os.path.join(C.OUT, "care_wt_pairs.csv")).set_index("patient")
cells = pairs.loc[M.index, ["tp_total", "r1_total"]].min(axis=1)
n = M.sum(axis=1)

order = cells.sort_values().index
Mo = M.loc[order]
co = cells.loc[order]

fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.6), gridspec_kw={"width_ratios": [1.45, 1]})

ax[0].imshow((~Mo).to_numpy(), aspect="auto", cmap="Greys", interpolation="nearest")
ax[0].set_xticks(range(M.shape[1]))
ax[0].set_xticklabels(M.columns, rotation=45, ha="right", fontsize=8)
ax[0].set_yticks([])
ax[0].set_ylabel("Patients (sorted by malignant cells recovered)", fontsize=8)
ax[0].set_title("a  Eligibility by state (20 cells at both endpoints)", fontsize=8, loc="left")

zero = int((n == 0).sum())
fig.text(0.012, 0.015, f"black = eligible; the {zero} patients above the broken line are "
         f"eligible for no state", fontsize=6.5, color="0.25")
ax[0].axhline(zero - 0.5, color="tab:red", lw=0.7, ls="--")

ax[1].scatter(cells, n, s=14, color="k", alpha=0.55, linewidth=0)
ax[1].set_xscale("log")
ax[1].set_xlabel("Malignant cells per pair (lower endpoint)", fontsize=8)
ax[1].set_ylabel("States eligible per patient", fontsize=8)
rho = spearmanr(cells, n)
ax[1].text(0.03, 0.95, f"\u03c1 = {rho.statistic:.2f}\nP = {rho.pvalue:.1e}", transform=ax[1].transAxes,
           fontsize=8, va="top")
ax[1].set_title("b  Eligibility clusters within patients", fontsize=8, loc="left")
for a in ax:
    a.tick_params(labelsize=7)
    for s in a.spines.values():
        s.set_linewidth(0.6)

fig.tight_layout()
out = os.path.join(C.OUT, "figures", "FigEligibility.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=400, bbox_inches="tight")
print("wrote", out)
