"""Figure 3: MGMT panel (eligibility rates, theta by rule, change by group)."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

cl = pd.read_excel(C.CARE_WT_CLINICAL, sheet_name="TableS1", header=2)
cl.columns = [str(c).strip() for c in cl.columns]
cl["Patient ID"] = cl["Patient ID"].astype(str).str.strip()
prim = cl[cl["Primary or recurrent"].astype(str).str.strip().str.lower().str.startswith("primary")]
mgmt = (prim.groupby("Patient ID")["MGMT protomoter methylation status"]
        .agg(lambda s: s.dropna().iloc[0] if s.notna().any() else None))

P = pd.read_csv(os.path.join(C.OUT, "care_wt_pairs.csv"))
P["g"] = P.patient.map(mgmt).astype(str).str.strip().str.lower()
K = P[P.g.isin(["methylated", "unmethylated"])].copy()
K["high"] = K.g == "methylated"
K["dMES"] = K.r1_frac - K.tp_frac
K["elig"] = (K.tp_MES >= C.MIN_CELLS) & (K.r1_MES >= C.MIN_CELLS)

fig, ax = plt.subplots(1, 3, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1, 1.25, 1.15]})

# a: eligibility by group
grp = K.groupby("high").elig.agg(["sum", "size"])
x = [0, 1]
okv = [grp.loc[False, "sum"], grp.loc[True, "sum"]]
nv = [grp.loc[False, "size"], grp.loc[True, "size"]]
ax[0].bar(x, nv, color="0.85", width=0.6)
ax[0].bar(x, okv, color="0.2", width=0.6)
for i, (o, n) in enumerate(zip(okv, nv)):
    ax[0].text(i, n + 0.6, f"{int(o)}/{int(n)}", ha="center", fontsize=7)
ax[0].set_xticks(x)
ax[0].set_xticklabels(["Unmethylated\n(n=24)", "Methylated\n(n=13)"], fontsize=7)
ax[0].set_ylabel("Strictly paired patients", fontsize=7)
ax[0].set_title("a  MES-eligible by MGMT status", fontsize=8, loc="left")
ax[0].set_ylim(0, max(nv) * 1.2)

# b: theta across rules, read from the frozen table
R = pd.read_csv(os.path.join(C.OUT, "mgmt_theta_by_rule.csv")).sort_values("rule")
rules = ["all" if r == 0 else str(r) for r in R.rule]
vals = [(r.theta, r.lo, r.hi) for _, r in R.iterrows()]
y = np.arange(len(rules))
for i, (v, lo, hi) in enumerate(vals):
    ax[1].plot([lo, hi], [i, i], color="0.3", lw=1.2)
    ax[1].plot(v, i, "o", color="k", ms=4)
ax[1].axvline(0, color="0.6", lw=0.8, ls=":")
ax[1].set_yticks(y)
ax[1].set_yticklabels([f">= {r} cells" if r != "all" else "all pairs" for r in rules], fontsize=7)
ax[1].set_xlabel("Median change difference (methylated - unmethylated)", fontsize=7)
ax[1].set_title("b  Same direction under every rule", fontsize=8, loc="left")
ax[1].invert_yaxis()

# c: distribution of within-patient change by group at the 20-cell rule
sub = K[K.elig]
for i, (lab, mask) in enumerate([("Unmeth.", ~sub.high), ("Meth.", sub.high)]):
    v = sub.loc[mask, "dMES"].to_numpy()
    ax[2].scatter(np.full(len(v), i) + np.random.default_rng(1).normal(0, 0.04, len(v)),
                  v, s=14, color="0.25", alpha=0.7, linewidth=0)
    ax[2].plot([i - 0.22, i + 0.22], [np.median(v)] * 2, color="tab:red", lw=1.4)
ax[2].axhline(0, color="0.6", lw=0.8, ls=":")
ax[2].set_xticks([0, 1])
ax[2].set_xticklabels(["Unmeth.", "Meth."], fontsize=7)
ax[2].set_ylabel("Change in MES-like fraction", fontsize=7)
ax[2].set_title("c  Eligible pairs only", fontsize=8, loc="left")

for a in ax:
    a.tick_params(labelsize=7)
    for s in a.spines.values():
        s.set_linewidth(0.6)
fig.tight_layout()
out = os.path.join(C.OUT, "figures", "FigMGMT.png")
fig.savefig(out, dpi=400, bbox_inches="tight")
print("wrote", out)
