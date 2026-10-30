"""
Probe: does the composition-channel structure reproduce in GSE174554
(IDH-wildtype GBM, 18 paired Primary->Recurrence patients, genome-wide
per-state pseudobulk)?

Design decisions frozen before running (see REPORT.md):
  - States with an archived pseudobulk score: MES-like, PN-like only.
    Intermediate has cell counts but NO pseudobulk column, so it cannot
    enter a scored decomposition.
  - Therefore the scored composition is restricted to MES+PN and weights are
    renormalised within that pair: w' = n_s / (n_MES + n_PN).
  - Outcome = change in MES share of scored cells, dw_MES = w'_MES,2 - w'_MES,1.
  - Expression: CPM per pseudobulk column, then log1p (frozen).
  - Two-state identity: dA = (wbar_M dx_M + wbar_P dx_P) + (xbar_M - xbar_P) * dw_M
  - Complete-case = MES and PN pseudobulk present at BOTH endpoints.
  - Sensitivity = impute the missing endpoint's state score by the other
    endpoint's value (dx = 0 for that state), which is what a pipeline that
    silently drops sub-threshold states effectively assumes.

No existing file is modified; everything is written under probe_GSE174554.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

SRC = C.DATA
PB = C.G174_EXPR
SUM = C.G174_META
OUT = C.OUT
os.makedirs(OUT, exist_ok=True)

IMPUTE = "--impute" in sys.argv
SUF = "_impute18" if IMPUTE else ""

LOX_FAMILY = ["LOX", "LOXL1", "LOXL2", "LOXL3", "LOXL4"]
MARKERS = ["CD44", "VIM", "FN1", "CHI3L1", "CHI3L2", "ANXA1", "TNC",
           "SPARC", "THBS2", "LTBP1", "PDGFRB", "TIMP3", "COL1A1", "COL3A1"]

# ----------------------------------------------------------------- load
pb = pd.read_csv(PB, sep="\t", index_col=0)
pb.index = pb.index.astype(str)
summ = pd.read_csv(SUM, sep="\t")
summ["sample_id"] = summ["sample_id"].astype(str)

lib = pb.sum(axis=0)
logcpm = np.log1p(pb.div(lib, axis=1) * 1e6)
print(f"pseudobulk: {pb.shape[0]} genes x {pb.shape[1]} state-columns")

# ------------------------------------------------- sample/state long table
recs = []
for col in pb.columns:
    sid, stname = col.split("|")
    row = summ.loc[summ.sample_id == sid].iloc[0]
    n_state = {"MES-like": int(row.MES_like_n), "PN-like": int(row.PN_like_n)}[stname]
    n_scored = int(row.MES_like_n) + int(row.PN_like_n)
    recs.append(dict(col=col, sample_id=sid, patient=row.patient_id,
                     status=row.status, state=stname, n_state=n_state,
                     n_scored=n_scored,
                     n_tumor=int(row.tumor_cells_total),
                     w_full=n_state / int(row.tumor_cells_total),
                     w_scored=n_state / n_scored))
meta = pd.DataFrame(recs)
meta.to_csv(os.path.join(OUT, "column_meta.tsv"), sep="\t", index=False)

# ------------------------------------------------------- patient matrices
patients = sorted(meta.patient.unique(), key=lambda x: int(x))
NX = {}
for pat in patients:
    sub = meta[meta.patient == pat]
    tp = dict(zip(sub.status, sub.sample_id))
    if "Primary" not in tp or "Recurrence" not in tp:
        continue
    NX[pat] = (tp["Primary"], tp["Recurrence"])

complete, incomplete = [], []
for pat, (s1, s2) in NX.items():
    c1 = set(meta[meta.sample_id == s1].state)
    c2 = set(meta[meta.sample_id == s2].state)
    (complete if {"MES-like", "PN-like"} <= c1 and
     {"MES-like", "PN-like"} <= c2 else incomplete).append(pat)

print(f"paired patients: {len(NX)}  complete-case: {len(complete)}  "
      f"dropped-state: {len(incomplete)} {incomplete}")

# ----------------------------------------------------------- decomposition
def decompose(pat, genes, impute_missing=False):
    """Return per-gene dA, within, comp, dMES for one patient."""
    s1, s2 = NX[pat]
    out = {}
    for t, sid in ((1, s1), (2, s2)):
        r = summ.loc[summ.sample_id == sid].iloc[0]
        ns = {"MES-like": int(r.MES_like_n), "PN-like": int(r.PN_like_n)}
        tot = ns["MES-like"] + ns["PN-like"]
        # weights come from cell counts, so they stay defined when a state's
        # pseudobulk column was dropped by the >=20-cell filter
        out[f"w{t}"] = {s: ns[s] / tot for s in ns}
        out[f"x{t}"] = {}
        for s in ("MES-like", "PN-like"):
            col = f"{sid}|{s}"
            out[f"x{t}"][s] = (logcpm.loc[genes, col].to_numpy()
                               if col in logcpm.columns else None)

    # impute a dropped state with the other endpoint's score (dx = 0)
    for s in ("MES-like", "PN-like"):
        if out["x1"][s] is None or out["x2"][s] is None:
            if not impute_missing:
                return None
            known = out["x2"][s] if out["x1"][s] is None else out["x1"][s]
            if known is None:      # state dropped at BOTH endpoints: not estimable
                return None
            if out["x1"][s] is None:
                out["x1"][s] = known
            else:
                out["x2"][s] = known

    w1M, w1P = out["w1"]["MES-like"], out["w1"]["PN-like"]
    w2M, w2P = out["w2"]["MES-like"], out["w2"]["PN-like"]
    x1M, x1P = out["x1"]["MES-like"], out["x1"]["PN-like"]
    x2M, x2P = out["x2"]["MES-like"], out["x2"]["PN-like"]

    A1 = w1M * x1M + w1P * x1P
    A2 = w2M * x2M + w2P * x2P
    dA = A2 - A1
    within = 0.5 * ((w1M + w2M) * (x2M - x1M) + (w1P + w2P) * (x2P - x1P))
    dwM = w2M - w1M
    comp = 0.5 * ((x1M + x2M) - (x1P + x2P)) * dwM
    resid = dA - (within + comp)
    return dict(dA=dA, within=within, comp=comp, dwM=dwM, resid=resid,
                D=(x1M + x2M) / 2 - (x1P + x2P) / 2)


# expressed-gene filter (frozen): CPM > 1 in >= 50% of state-columns
cpm = pb.div(lib, axis=1) * 1e6
keep = (cpm > 1).mean(axis=1) >= 0.5
genes = pb.index[keep].to_numpy()
print(f"genes passing expression filter: {len(genes)} / {pb.shape[0]}")

rows, pats_used = [], complete
if IMPUTE:
    res, pats_used = {}, list(NX)
    for p in pats_used:
        r = decompose(p, genes, impute_missing=True)
        if r is not None:
            res[p] = r
    pats_used = [p for p in pats_used if p in res]
else:
    res = {p: decompose(p, genes) for p in pats_used}
res = {p: v for p, v in res.items() if v is not None}
pats_used = list(res)

max_resid = max(float(np.nanmax(np.abs(res[p]["resid"]))) for p in pats_used)
print(f"max identity residual over all genes/patients: {max_resid:.3e}")

dMES = np.array([res[p]["dwM"] for p in pats_used])
for i, g in enumerate(genes):
    dA = np.array([res[p]["dA"][i] for p in pats_used])
    wi = np.array([res[p]["within"][i] for p in pats_used])
    co = np.array([res[p]["comp"][i] for p in pats_used])
    Dv = np.array([res[p]["D"][i] for p in pats_used])
    if np.nanstd(dA) == 0:
        continue
    rows.append(dict(
        gene=g, n=len(pats_used),
        rho_dA=stats.spearmanr(dA, dMES).statistic,
        rho_within=stats.spearmanr(wi, dMES).statistic,
        rho_comp=stats.spearmanr(co, dMES).statistic,
        D_median=float(np.nanmedian(Dv)),
        D_frac_pos=float(np.nanmean(Dv > 0)),
        med_abs_within=float(np.nanmedian(np.abs(wi))),
        med_abs_comp=float(np.nanmedian(np.abs(co))),
    ))

gdf = pd.DataFrame(rows)
gdf["within_share"] = gdf.med_abs_within / (gdf.med_abs_within + gdf.med_abs_comp)
gdf["channel"] = np.where(gdf.rho_comp.abs() > gdf.rho_within.abs(),
                          "composition", "within")
gdf.to_csv(os.path.join(OUT, f"gene_decomposition_GSE174554{SUF}.tsv"),
           sep="\t", index=False)

# ------------------------------------------------------------ LOX family
focus = gdf[gdf.gene.isin(LOX_FAMILY + MARKERS)].copy()
pct = {g: (gdf.rho_dA < r.rho_dA).mean() for g, r in
       focus.set_index("gene").iterrows()}
focus["rho_dA_percentile"] = focus.gene.map(pct)
focus.sort_values("rho_dA", ascending=False).to_csv(
    os.path.join(OUT, f"lox_family_and_markers{SUF}.tsv"), sep="\t", index=False)

fam = []
for g in LOX_FAMILY:
    if g not in pb.index:
        fam.append(dict(gene=g, status="absent_from_matrix"))
    else:
        fam.append(dict(gene=g,
                        status="tested" if g in set(genes) else "filtered_low_expression",
                        mean_cpm=float(cpm.loc[g].mean()),
                        frac_cols_cpm_gt1=float((cpm.loc[g] > 1).mean())))
fam = pd.DataFrame(fam)
fam.to_csv(os.path.join(OUT, f"lox_family_expression_status{SUF}.tsv"),
           sep="\t", index=False)
print("\n--- LOX family expression status ---")
print(fam.round(3).to_string(index=False))
print("\n--- LOX family / MES markers (ranked by rho_dA) ---")
print(focus.sort_values("rho_dA", ascending=False)[
    ["gene", "rho_dA", "rho_within", "rho_comp", "D_median",
     "within_share", "channel", "rho_dA_percentile"]].to_string(index=False))

# ------------------------------------------------ genome-wide channel map
sig = gdf[gdf.rho_dA.abs() >= 0.5]
summary = {
    "n_patients_paired": int(len(NX)),
    "n_complete_case": int(len(pats_used)),
    "dropped_state_patients": [int(x) for x in incomplete],
    "n_genes_tested": int(len(gdf)),
    "identity_max_abs_residual": max_resid,
    "median_within_share_genomewide": float(gdf.within_share.median()),
    "frac_genes_composition_channel": float((gdf.channel == "composition").mean()),
    "n_genes_rho_dA_ge_0.5": int(len(sig)),
    "frac_composition_channel_among_strong": float((sig.channel == "composition").mean())
    if len(sig) else None,
    "median_rho_within": float(gdf.rho_within.median()),
    "median_rho_comp": float(gdf.rho_comp.median()),
    "mediate_corr": float(stats.spearmanr(gdf.rho_within, gdf.rho_comp).statistic),
}
with open(os.path.join(OUT, f"channel_summary{SUF}.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# ------------------------------------------------- filter bias (real data)
fb = []
for pat, (s1, s2) in NX.items():
    row = {"patient": pat}
    for t, sid in ((1, s1), (2, s2)):
        r = summ.loc[summ.sample_id == sid].iloc[0]
        hasPN = f"{sid}|PN-like" in pb.columns
        hasMES = f"{sid}|MES-like" in pb.columns
        true_M = r.MES_like_fraction
        # what the pipeline sees: observed states only, renormalised
        obs = (int(r.MES_like_n) if hasMES else 0) + (int(r.PN_like_n) if hasPN else 0)
        row[f"w_true_t{t}"] = true_M
        row[f"w_pipe_t{t}"] = (int(r.MES_like_n) / obs) if obs else np.nan
        row[f"PN_dropped_t{t}"] = not hasPN
    row["dw_true"] = row["w_true_t2"] - row["w_true_t1"]
    row["dw_pipe"] = row["w_pipe_t2"] - row["w_pipe_t1"]
    row["sign_flip"] = np.sign(row["dw_true"]) != np.sign(row["dw_pipe"])
    fb.append(row)
fb = pd.DataFrame(fb)
fb.to_csv(os.path.join(OUT, f"filter_bias_realdata{SUF}.tsv"), sep="\t", index=False)
summary["filter_n_dropped_pn_slots"] = int(fb[["PN_dropped_t1", "PN_dropped_t2"]]
                                           .sum().sum())
summary["filter_n_sign_flips_dw"] = int(fb.sign_flip.sum())
summary["filter_max_abs_dw_shift"] = float(
    (fb.dw_pipe - fb.dw_true).abs().max())

with open(os.path.join(OUT, f"channel_summary{SUF}.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n--- filter bias (real data) ---")
print(fb[["patient", "PN_dropped_t1", "PN_dropped_t2", "w_true_t1", "w_pipe_t1",
          "w_true_t2", "w_pipe_t2", "dw_true", "dw_pipe", "sign_flip"]]
      .round(4).to_string(index=False))

print("\n--- summary ---")
print(json.dumps(summary, indent=2, ensure_ascii=False))

# ------------------------------------------- CARE comparison reference
print("\nCARE reference (from JNO remediation / v2.2): "
      "rho_within=0.35 / 0.57, rho_comp=0.63 / 0.86, "
      "within magnitude share 71% / 63%")
