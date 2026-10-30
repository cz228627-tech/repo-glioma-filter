"""
Composition-corrected within-state differential change, GSE174554.

Question: after removing the composition channel, what actually changes INSIDE
malignant MES-like and PN-like cells between primary and recurrence?

This is the estimator use of the decomposition (not the diagnostic use):
  dx_s = x_s,recurrence - x_s,primary  per patient, per state
  paired Wilcoxon signed-rank across patients, BH-FDR per state
Only patients with the state present at both endpoints enter (complete case).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
import os

import numpy as np
import pandas as pd
from scipy import stats

SRC = C.DATA
PB = C.G174_EXPR
SUM = C.G174_META
OUT = C.OUT

pb = pd.read_csv(PB, sep="\t", index_col=0)
summ = pd.read_csv(SUM, sep="\t")
logcpm = np.log1p(pb.div(pb.sum(axis=0), axis=1) * 1e6)
cpm = pb.div(pb.sum(axis=0), axis=1) * 1e6
keep = (cpm > 1).mean(axis=1) >= 0.5
genes = pb.index[keep].to_numpy()

pairs = []
for pat, g in summ.groupby("patient_id"):
    tp = dict(zip(g.status, g.sample_id))
    if "Primary" in tp and "Recurrence" in tp:
        pairs.append((pat, tp["Primary"], tp["Recurrence"]))

rows = []
for state in ("MES-like", "PN-like"):
    D = []
    used = []
    for pat, s1, s2 in pairs:
        c1, c2 = f"{s1}|{state}", f"{s2}|{state}"
        if c1 in logcpm.columns and c2 in logcpm.columns:
            D.append((logcpm[c2] - logcpm[c1]).loc[genes].to_numpy())
            used.append(pat)
    if len(D) < 6:
        continue
    D = np.vstack(D)
    n = len(D)
    for i, gene in enumerate(genes):
        d = D[:, i]
        if np.all(d == 0):
            continue
        try:
            p = stats.wilcoxon(d, zero_method="wilcox").pvalue
        except ValueError:
            p = 1.0
        rows.append(dict(state=state, gene=gene, n=n,
                         mean_delta=float(d.mean()), median_delta=float(np.median(d)),
                         frac_up=float((d > 0).mean()), p=p))

df = pd.DataFrame(rows)
for state, sub in df.groupby("state"):
    idx = sub.index
    q = stats.false_discovery_control(sub.p.values, method="bh")
    df.loc[idx, "q"] = q
df.to_csv(os.path.join(OUT, "within_state_de_GSE174554.tsv"), sep="\t", index=False)

for state, sub in df.groupby("state"):
    print(f"\n===== {state} : n={sub.n.iloc[0]} patients, {len(sub)} genes =====")
    print(f"q<0.05: {(sub.q < 0.05).sum()}   q<0.01: {(sub.q < 0.01).sum()}")
    top = sub.sort_values("q").head(15)
    print(top[["gene", "n", "mean_delta", "frac_up", "p", "q"]].round(4).to_string(index=False))

# LOX-family / ECM panel check inside the same framework
panel = ["LOX", "LOXL1", "LOXL2", "LOXL3", "LOXL4", "CD44", "VIM", "FN1",
         "CHI3L1", "ANXA1", "SPARC", "POSTN", "COL6A1", "COL11A1", "TYRP1"]
print("\n===== panel genes in the within-state DE =====")
pt = df[df.gene.isin(panel)].pivot_table(index="gene", columns="state",
                                         values=["mean_delta", "q"])
print(pt.round(4).to_string())
