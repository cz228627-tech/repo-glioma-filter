"""
Six-state composition decomposition of the frozen CARE / GSE292130 62-gene
panel tables (local copies in C:\\Users\\86183\\Downloads).

No download is needed: the frozen inputs already exist.
  - <cohort>_ecm_gene_expression.tsv : 209 (CARE) rows x 62 genes, row = pb_group
  - <cohort>_custom_pseudobulk.tsv   : pb_group, patient, timepoint, cell_state,
                                       n_cells, module scores

Design (frozen before running):
  - endpoints: CARE -> timepoint 1 -> 2 (T3 unused, per manuscript);
               GSE292130 -> earliest -> latest per patient
  - w_{s,t} = n_cells / sum_s n_cells at that patient-timepoint (pipeline view:
              only states that survived the >=20-cell filter are seen)
  - A_t = sum_s w_{s,t} x_{s,t}; x taken as archived (already log-scale)
  - states present at both endpoints: within = wbar*dx, comp = xbar*dw
  - states present at one endpoint only: exact turnover contribution
  - outcome: dw_MES = w_MES,2 - w_MES,1
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

SRC = C.DATA
OUT = C.OUT
os.makedirs(OUT, exist_ok=True)

COHORTS = {
    "CARE": ("gse326221_ecm_gene_expression.tsv", "gse326221_custom_pseudobulk.tsv"),
    "GSE292130": ("gse292130_ecm_gene_expression.tsv", "gse292130_custom_pseudobulk.tsv"),
}

summary = {}
for cohort, (ef, pf) in COHORTS.items():
    expr = pd.read_csv(os.path.join(SRC, ef), sep="\t", index_col=0)
    meta = pd.read_csv(os.path.join(SRC, pf), sep="\t").set_index("pb_group")
    meta = meta.loc[expr.index]
    genes = list(expr.columns)
    X = expr.to_numpy(float)
    meta = meta.reset_index().rename(columns={"index": "pb_group"})
    if "timepoint" not in meta.columns:
        meta["timepoint"] = meta["timepoint_num"]

    patients = sorted(meta.patient.unique())
    rows, used = [], []
    max_resid = 0.0
    for pat in patients:
        sub = meta[meta.patient == pat]
        tps = sorted(sub.timepoint.unique())
        if len(tps) < 2:
            continue
        t1, t2 = tps[0], tps[-1]
        d1 = sub[sub.timepoint == t1].set_index("cell_state", drop=False)
        d2 = sub[sub.timepoint == t2].set_index("cell_state", drop=False)

        n1 = d1.n_cells.astype(float)
        n2 = d2.n_cells.astype(float)
        w1 = n1 / n1.sum()
        w2 = n2 / n2.sum()

        def x_of(d, s):
            return X[list(expr.index).index(d.loc[s, "pb_group"])] if s in d.index else None

        within = np.zeros(len(genes))
        comp = np.zeros(len(genes))
        turn = np.zeros(len(genes))
        A1 = np.zeros(len(genes))
        A2 = np.zeros(len(genes))
        for s in set(d1.index) | set(d2.index):
            x1, x2 = x_of(d1, s), x_of(d2, s)
            if x1 is not None and x2 is not None:
                a, b = w1[s], w2[s]
                within += 0.5 * (a + b) * (x2 - x1)
                comp += 0.5 * (x1 + x2) * (b - a)
                A1 += a * x1
                A2 += b * x2
            elif x1 is not None:      # state vanished
                turn -= w1[s] * x1
                A1 += w1[s] * x1
            else:                     # state appeared
                turn += w2[s] * x2
                A2 += w2[s] * x2

        dA = A2 - A1
        resid = np.abs(dA - (within + comp + turn)).max()
        max_resid = max(max_resid, resid)
        dwM = float(w2.get("MES_like", 0.0) - w1.get("MES_like", 0.0))
        used.append(dict(patient=pat, dMES=dwM, n_states_t1=len(d1),
                         n_states_t2=len(d2), resid=resid))
        for i, g in enumerate(genes):
            rows.append(dict(patient=pat, gene=g, dA=dA[i], within=within[i],
                             comp=comp[i], turnover=turn[i], dMES=dwM))

    df = pd.DataFrame(rows)
    dM = df.groupby("patient").dMES.first()
    out = []
    for g, gd in df.groupby("gene"):
        gd = gd.set_index("patient").loc[dM.index]
        dA, wi, co = gd.dA.values, gd.within.values, gd.comp.values
        if np.std(dA) == 0:
            continue
        out.append(dict(gene=g, n=len(dA),
                        rho_dA=stats.spearmanr(dA, dM.values).statistic,
                        rho_within=stats.spearmanr(wi, dM.values).statistic,
                        rho_comp=stats.spearmanr(co, dM.values).statistic,
                        med_abs_within=float(np.median(np.abs(wi))),
                        med_abs_comp=float(np.median(np.abs(co))),
                        med_abs_turn=float(np.median(np.abs(gd.turnover.values)))))
    gd = pd.DataFrame(out)
    gd["within_share"] = gd.med_abs_within / (gd.med_abs_within + gd.med_abs_comp)
    gd["channel"] = np.where(gd.rho_comp.abs() > gd.rho_within.abs(),
                             "composition", "within")
    gd["rank_rho_dA"] = gd.rho_dA.rank(ascending=False).astype(int)
    gd.sort_values("rho_dA", ascending=False).to_csv(
        os.path.join(OUT, f"gene_decomposition_{cohort}.tsv"), sep="\t",
        index=False)

    u = pd.DataFrame(used)
    summary[cohort] = dict(
        n_patients=len(u), n_genes=len(gd),
        identity_max_abs_residual=float(max_resid),
        median_within_share=float(gd.within_share.median()),
        frac_composition_channel=float((gd.channel == "composition").mean()),
        median_rho_within=float(gd.rho_within.median()),
        median_rho_comp=float(gd.rho_comp.median()),
        patients_with_turnover=int((u.n_states_t1 != u.n_states_t2).sum()),
    )
    print(f"\n===== {cohort} =====")
    print(json.dumps(summary[cohort], indent=2, ensure_ascii=False))
    show = gd[gd.gene.isin(["LOX", "LOXL1", "LOXL2", "LOXL3", "LOXL4", "CD44",
                            "CHI3L1", "FN1", "VIM", "ANXA1", "SPARC", "TNC"])]
    print(show[["gene", "rank_rho_dA", "rho_dA", "rho_within", "rho_comp",
                "within_share", "channel"]].round(3).to_string(index=False))

with open(os.path.join(OUT, "six_state_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
