"""Reproduce Figures 1-3 from the derived tables."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

OUT = os.path.join(C.OUT, "figures")
os.makedirs(OUT, exist_ok=True)
RES = C.OUT
BLUE, RED, GREEN, GREY = "#2f6f9f", "#c8553d", "#6b8f71", "#8a8a8a"

# ------------------------------------------------------------------ data
X = pd.read_csv(C.CARE_EXPR, sep="\t", index_col=0)
m = pd.read_csv(C.CARE_META, sep="\t").set_index("pb_group")
m = m.loc[X.index].reset_index().rename(columns={"index": "pb_group"})

PAT, GENE = 61, "FN1"
s = m[m.patient == PAT]
tps = sorted(s.timepoint.unique())
d1 = s[s.timepoint == tps[0]].set_index("cell_state")
d2 = s[s.timepoint == tps[-1]].set_index("cell_state")
w1 = d1.n_cells / d1.n_cells.sum()
w2 = d2.n_cells / d2.n_cells.sum()
order = ["MES_like", "AC_like", "OPC_like", "NPC_like", "Cycling", "Undifferentiated"]
label = {"MES_like": "MES-like", "AC_like": "AC-like", "OPC_like": "OPC-like",
         "NPC_like": "NPC-like", "Cycling": "Cycling", "Undifferentiated": "Undiff."}
rows = []
for st in order:
    i1, i2 = st in d1.index, st in d2.index
    v1 = X.loc[d1.loc[st, "pb_group"], GENE] if i1 else np.nan
    v2 = X.loc[d2.loc[st, "pb_group"], GENE] if i2 else np.nan
    a, b = (w1[st] if i1 else 0.0), (w2[st] if i2 else 0.0)
    if i1 and i2:
        wi, co, tu = 0.5 * (a + b) * (v2 - v1), 0.5 * (v1 + v2) * (b - a), 0.0
    elif i1:
        wi = co = 0.0; tu = -a * v1
    else:
        wi = co = 0.0; tu = b * v2
    rows.append(dict(state=label[st], w1=a, w2=b, x1=v1, x2=v2, within=wi, comp=co, turn=tu))
dfT = pd.DataFrame(rows)
wi_t, co_t, tu_t = dfT.within.sum(), dfT.comp.sum(), dfT.turn.sum()
dA = wi_t + co_t + tu_t

# eligibility + distortion (audited 52-pair set)
P = pd.read_csv(os.path.join(C.OUT, "care_wt_pairs.csv"))
P["retained"] = (P.tp_MES >= C.MIN_CELLS) & (P.r1_MES >= C.MIN_CELLS)
D = pd.read_csv(os.path.join(C.OUT, "care_wt_distortion.csv"))
D = D.rename(columns={"dw_all": "dw_true", "dw_renorm": "dw_pipe"})

# ================================================================== FIG 1
fig = plt.figure(figsize=(11.0, 7.2), dpi=400)
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.95], hspace=0.30, wspace=0.30)

axa = fig.add_subplot(gs[0, 0]); axa.axis("off"); axa.set_xlim(0, 1); axa.set_ylim(0, 1)
axa.text(0, 1.02, "a", fontsize=12, fontweight="bold", va="top")
axa.text(0.02, 0.95, r"$A_t=\sum_s w_{s,t}\,x_{s,t}$", fontsize=11.5, va="top")
axa.text(0.02, 0.80, r"$\Delta A=\sum_s \bar w_s\,\Delta x_s+\sum_s \bar x_s\,\Delta w_s+T$",
         fontsize=10.5, va="top")
for y, col, term, name in [(0.60, BLUE, r"$\sum_s \bar w_s\,\Delta x_s$", "within-state"),
                           (0.48, RED, r"$\sum_s \bar x_s\,\Delta w_s$", "composition"),
                           (0.36, GREY, r"$T$", "turnover")]:
    axa.add_patch(plt.Rectangle((0.02, y + 0.01), 0.022, 0.03, color=col))
    axa.text(0.05, y + 0.025, term, fontsize=9.0, va="center", color=col)
    axa.text(0.40, y + 0.025, name, fontsize=7.8, va="center", color="0.25")
axa.text(0.02, 0.26, "w = state weight (fraction of cells)\nx = state-level expression",
         fontsize=7.6, va="top", color="0.3")
axa.text(0.02, 0.12, r"residual $\leq 2.3\times10^{-15}$", fontsize=7.2, va="top", color="0.45")

axb = fig.add_subplot(gs[0, 1:]); axb.axis("off"); axb.set_xlim(0, 1); axb.set_ylim(0, 1)
axb.text(0, 1.02, "b", fontsize=12, fontweight="bold", va="top")
axb.text(0.06, 1.02, f"patient {PAT}, {GENE}", fontsize=9, va="top")
cols = ["state", "w$_{T1}$", "w$_{T2}$", "x$_{T1}$", "x$_{T2}$", "within", "comp.", "turn."]
xpos = [0.05, 0.22, 0.33, 0.42, 0.51, 0.62, 0.76, 0.89]
top = 0.86
for j, c in enumerate(cols):
    axb.text(xpos[j], top, c, fontsize=7.6, fontweight="bold")
axb.plot([0.03, 0.99], [top - 0.04, top - 0.04], color="0.65", lw=0.7)
for i, r in dfT.iterrows():
    y = top - 0.115 - i * 0.108
    if r.state == "MES-like":
        axb.add_patch(plt.Rectangle((0.02, y - 0.03), 0.97, 0.09, color="#fdece0", zorder=0))
    vals = [r.state, f"{r.w1:.3f}", f"{r.w2:.3f}",
            ("absent" if np.isnan(r.x1) else f"{r.x1:.2f}"),
            ("absent" if np.isnan(r.x2) else f"{r.x2:.2f}"),
            f"{r.within:+.3f}", f"{r.comp:+.3f}", f"{r.turn:+.3f}"]
    for j, v in enumerate(vals):
        axb.text(xpos[j], y, v, fontsize=(6.4 if v == "absent" else 8.0),
                 color=("0.55" if v == "absent" else "0.1"))
yb = top - 0.115 - len(dfT) * 0.108
axb.plot([0.03, 0.99], [yb + 0.05, yb + 0.05], color="0.65", lw=0.7)
for j, v in enumerate(["total", "1.000", "1.000", "", "", f"{wi_t:+.3f}", f"{co_t:+.3f}", f"{tu_t:+.3f}"]):
    axb.text(xpos[j], yb, v, fontsize=7.4, fontweight="bold")

sets = [("CARE\n(6 states)", "gene_decomposition_CARE.tsv", BLUE),
        ("GSE292130\n(6 states)", "gene_decomposition_GSE292130.tsv", RED),
        ("GSE174554\n(2 states)", "gene_decomposition_GSE174554.tsv", GREEN)]
for k, (lab, f, col) in enumerate(sets):
    ax = fig.add_subplot(gs[1, k])
    d = pd.read_csv(os.path.join(RES, f), sep="\t")
    v = d.within_share.dropna().values
    parts = ax.violinplot([v], positions=[0], widths=0.7, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(col); pc.set_alpha(0.35); pc.set_edgecolor("none")
    if len(v) <= 100:
        ax.scatter(np.random.default_rng(1).normal(0, 0.05, len(v)), v, s=8,
                   color=col, alpha=0.7, edgecolor="none", zorder=3)
    ax.axhline(np.median(v), color=col, ls="--", lw=1.3)
    ax.text(0.5, np.median(v) + 0.04, f"{np.median(v):.3f}", fontsize=7.2, color=col, ha="right")
    ax.set_xlim(-0.6, 0.6); ax.set_ylim(0, 1.02); ax.set_xticks([])
    ax.set_title(lab, fontsize=8)
    if k == 0:
        ax.set_ylabel("within-state share\nof magnitude", fontsize=7.6)
    if k == 0:
        ax.text(-0.32, 1.12, "c", fontsize=12, fontweight="bold", transform=ax.transAxes)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
fig.savefig(os.path.join(OUT, "Fig1.png"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "Fig1.pdf"), bbox_inches="tight")
plt.close(fig)

# ================================================================== FIG 2
both = int(((P.tp_MES >= 20) & (P.r1_MES >= 20)).sum())
tp_only = int(((P.tp_MES < 20) & (P.r1_MES >= 20)).sum())
r1_only = int(((P.tp_MES >= 20) & (P.r1_MES < 20)).sum())
neither = int(((P.tp_MES < 20) & (P.r1_MES < 20)).sum())
pmc = binomtest(r1_only, r1_only + tp_only).pvalue
ths = [0, 5, 10, 20, 50, 100]
kept = [int(((P.tp_MES >= t) & (P.r1_MES >= t)).sum()) for t in ths]

fig, ax = plt.subplots(3, 2, figsize=(9.2, 10.4), dpi=400)
(axa, axb), (axc, axd), (axe, axf) = ax
P["retained"] = (P.tp_MES >= 20) & (P.r1_MES >= 20)
axa.plot(range(len(ths)), kept, "-o", color=BLUE, lw=1.6, ms=6)
for i, k in enumerate(kept):
    axa.annotate(str(k), (i, k), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
axa.axhline(len(P), color="0.75", ls=":", lw=1)
axa.text(len(ths) - 1, len(P) + 0.8, f"all strictly paired (n = {len(P)})", fontsize=7.2,
         color="0.45", ha="right")
axa.axvline(3, color=RED, ls="--", lw=1)
axa.text(3.1, 10, "20-cell\nthreshold", fontsize=7.2, color=RED)
axa.set_xticks(range(len(ths))); axa.set_xticklabels([f"$\\geq${t}" for t in ths], fontsize=8)
axa.set_xlabel("minimum MES-like cells at both endpoints", fontsize=8)
axa.set_ylabel("patients retained", fontsize=8); axa.set_ylim(0, len(P) + 5)
axa.set_title("a", fontsize=11, fontweight="bold", loc="left")
for sp in ("top", "right"): axa.spines[sp].set_visible(False)

cats = ["both\npass", "TP\nonly", "R1\nonly", "both\nfail"]
vals = [both, tp_only, r1_only, neither]
axb.bar(range(4), vals, color=[BLUE, "0.75", RED, "0.45"], width=0.62, edgecolor="white")
for i, v in enumerate(vals):
    axb.text(i, v + 0.6, str(v), ha="center", fontsize=9, fontweight="bold")
axb.set_xticks(range(4)); axb.set_xticklabels(cats, fontsize=8)
axb.set_ylabel(f"patients (of {len(P)})", fontsize=8); axb.set_ylim(0, max(vals) * 1.25)
axb.text(0.5, 0.92, f"exact McNemar $P$ = {pmc:.3f}", transform=axb.transAxes,
         fontsize=8, color="0.3", ha="center")
axb.set_title("b", fontsize=11, fontweight="bold", loc="left")
for sp in ("top", "right"): axb.spines[sp].set_visible(False)

axc.scatter(P.tp_total, P.r1_total, s=22, color=BLUE, alpha=0.75, edgecolor="white", linewidth=0.4)
lim = [min(P.tp_total.min(), P.r1_total.min()) * 0.8, max(P.tp_total.max(), P.r1_total.max()) * 1.2]
axc.plot(lim, lim, ls="--", color="0.6", lw=0.9)
axc.set_xscale("log"); axc.set_yscale("log"); axc.set_xlim(lim); axc.set_ylim(lim)
axc.set_xlabel("malignant cells, primary", fontsize=8)
axc.set_ylabel("malignant cells, recurrence", fontsize=8)
axc.text(0.04, 0.96, f"medians: {P.tp_total.median():.0f} vs {P.r1_total.median():.0f}",
         transform=axc.transAxes, fontsize=7.2, va="top", color="0.3")
axc.set_title("c", fontsize=11, fontweight="bold", loc="left")
for sp in ("top", "right"): axc.spines[sp].set_visible(False)

axd.scatter(D.dw_true, D.dw_pipe, s=26, color=GREEN, alpha=0.85, edgecolor="white", linewidth=0.4)
lim2 = [min(D.dw_true.min(), D.dw_pipe.min()) - 0.03, max(D.dw_true.max(), D.dw_pipe.max()) + 0.03]
axd.plot(lim2, lim2, ls="--", color="0.6", lw=0.9)
axd.set_xlim(lim2); axd.set_ylim(lim2)
axd.axhline(0, color="0.85", lw=0.6); axd.axvline(0, color="0.85", lw=0.6)
axd.set_xlabel("MES weight change, all cells", fontsize=8)
axd.set_ylabel("after dropping states $<20$ cells", fontsize=8)
r = np.corrcoef(D.dw_true, D.dw_pipe)[0, 1]
axd.text(0.04, 0.96, f"n = {len(D)}\nmedian |diff| = {np.median(np.abs(D.dw_pipe - D.dw_true)):.4f}\n"
                     f"r = {r:.2f}", transform=axd.transAxes, fontsize=7.2, va="top", color="0.3")
axd.set_title("d", fontsize=11, fontweight="bold", loc="left")
for sp in ("top", "right"): axd.spines[sp].set_visible(False)

rng = np.random.default_rng(3)
for k, (grp, col) in enumerate([(True, BLUE), (False, RED)]):
    v = P[P.retained == grp]
    axe.scatter(np.full(len(v), k) + rng.normal(0, 0.05, len(v)), v.tp_total,
                s=22, color=col, alpha=0.8, edgecolor="white", linewidth=0.4)
    axf.scatter(np.full(len(v), k) + rng.normal(0, 0.05, len(v)), v.r1_frac,
                s=22, color=col, alpha=0.8, edgecolor="white", linewidth=0.4)
axe.set_yscale("log")
axe.set_xticks([0, 1]); axe.set_xticklabels([f"retained\n(n={int(P.retained.sum())})",
                                            f"excluded\n(n={int((~P.retained).sum())})"], fontsize=8)
axe.set_ylabel("malignant cells, primary", fontsize=8)
axe.set_title("e", fontsize=11, fontweight="bold", loc="left")
axf.set_xticks([0, 1]); axf.set_xticklabels([f"retained\n(n={int(P.retained.sum())})",
                                            f"excluded\n(n={int((~P.retained).sum())})"], fontsize=8)
axf.set_ylabel("MES fraction, recurrence", fontsize=8)
axf.set_title("f", fontsize=11, fontweight="bold", loc="left")
for a_ in (axe, axf):
    for sp in ("top", "right"): a_.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "Fig2.png"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "Fig2.pdf"), bbox_inches="tight")
plt.close(fig)

# ================================================================== FIG 3
b = pd.read_csv(os.path.join(RES, "P0_1_corrected_benchmark.tsv"), sep="\t")
lab = {"S1 no composition change": "S1\nno comp.\nchange", "S3 both +offset": "S3\nboth\n+offset",
       "S6 filter on": "S6\nfilter", "S7 filter, rare MES": "S7\nrare\n+ filter"}
sel = [k for k in lab if k in set(b.scenario)]
b = b[b.scenario.isin(sel)].set_index("scenario").loc[sel]
fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.8), dpi=400, gridspec_kw=dict(wspace=0.42,
                                                                            width_ratios=[1.15, 1.1, 1.0]))
a0, a1, a2 = ax
x = np.arange(len(sel)); w = 0.26
a0.bar(x - w, b.E1_signerr, w, color=BLUE, label="decomposition (contribution)")
a0.bar(x, b.E3_signerr, w, color=RED, label="covariate-adjusted (contribution)")
a0.bar(x + w, b.E2_signerr, w, color=GREEN, hatch="///", edgecolor="white", label="state-level (effect)")
a0.set_xticks(x); a0.set_xticklabels([lab[s] for s in sel], fontsize=6.8)
a0.set_ylabel("sign-error rate", fontsize=8); a0.set_ylim(0, 0.65)
a0.legend(fontsize=5.6, frameon=False, loc="upper left")
a0.set_title("a", fontsize=11, fontweight="bold", loc="left")
for sp in ("top", "right"): a0.spines[sp].set_visible(False)

a1.bar(x - 0.17, b.E1_rmse, 0.32, color=BLUE, label="decomposition")
a1.bar(x + 0.17, b.E3_rmse, 0.32, color=RED, label="covariate-adjusted")
a1.set_xticks(x); a1.set_xticklabels([lab[s] for s in sel], fontsize=6.8)
a1.set_ylabel("RMSE (same target)", fontsize=8); a1.set_ylim(0, 0.075)
a1.legend(fontsize=5.8, frameon=False, loc="upper left")
a1.set_title("b", fontsize=11, fontweight="bold", loc="left")
for sp in ("top", "right"): a1.spines[sp].set_visible(False)

mg = 16044
ns = np.arange(8, 31)
k = np.ceil(mg * 2.0 ** (1 - ns) / 0.05)
a2.plot(ns, k, "-o", ms=3.2, lw=1.3, color="0.35")
a2.set_yscale("log")
for n0 in (14, 18):
    k0 = np.ceil(mg * 2.0 ** (1 - n0) / 0.05)
    a2.plot([n0], [k0], "o", ms=6, mfc="none", mec=RED, mew=1.4)
    a2.annotate(f"n={n0}: {int(k0)}", (n0, k0), textcoords="offset points", xytext=(8, 6),
                fontsize=6.8, color=RED)
a2.set_xlabel("paired patients (n)", fontsize=8)
a2.set_ylabel("genes at the P floor\nneeded for BH q<0.05", fontsize=7.4)
a2.set_title("c", fontsize=11, fontweight="bold", loc="left")
for sp in ("top", "right"): a2.spines[sp].set_visible(False)
fig.savefig(os.path.join(OUT, "Fig3.png"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "Fig3.pdf"), bbox_inches="tight")
plt.close(fig)

print("pairs", len(P), "dist", len(D), "both", both, tp_only, r1_only, neither)
print("median|diff|", round(np.median(np.abs(D.dw_pipe - D.dw_true)), 4), "r", round(r, 3))
print("saved figs to", OUT)


