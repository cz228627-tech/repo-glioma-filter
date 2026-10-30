"""CARE wild-type cohort: eligibility under a cell-count threshold, and how far
renormalisation moves the composition estimate for the patients retained.

Inputs
  data/41588_2025_2168_MOESM2_ESM.xlsx   Supplementary Table 1 of Spitzer et al.
                                         (Nat Genet 2025); specimen-level labels
  data/care_wt_state_counts.csv          malignant cells per sample x state,
                                         produced by scripts/dump_counts.R
  data/care_wt_synapse_dna_snrna_id_linker.txt  aliquot identifiers (used only
                                         to document label disagreements)

Outputs (outputs/)
  care_wt_pairs.csv          one row per strictly paired patient
  care_wt_thresholds.csv     patients retained by each minimum cell count
  care_wt_between_group.csv  retained vs excluded comparison
  care_wt_distortion.csv     renormalisation effect among retained patients
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

rng = np.random.default_rng(C.SEED)


def load_clinical():
    cl = pd.read_excel(C.CARE_WT_CLINICAL, sheet_name="TableS1", header=2).dropna(how="all")
    cl.columns = [str(c).strip() for c in cl.columns]
    for c in ("Patient ID", "Time point", "Primary or recurrent", "Sample ID"):
        cl[c] = cl[c].astype(str).str.strip()
    return cl


def build_pairs(cl, counts):
    """Strict pairs follow the published clinical annotation, not sample names."""
    idx = cl.set_index(["Patient ID", "Time point"])
    tot = counts.groupby("ID").n.sum()
    mes = counts[counts.State == "MES"].set_index("ID").n

    rows, dropped = [], []
    for pid in sorted(cl["Patient ID"].unique(), key=lambda x: int(x[1:])):
        if (pid, "T1") not in idx.index or (pid, "T2") not in idx.index:
            dropped.append((pid, "single time point"))
            continue
        s1, s2 = idx.loc[(pid, "T1")], idx.loc[(pid, "T2")]
        if s1["Primary or recurrent"] != "Primary" or s2["Primary or recurrent"] != "1st Recurrent":
            dropped.append((pid, f"{s1['Primary or recurrent']} -> {s2['Primary or recurrent']}"))
            continue
        t1, t2 = s1["Sample ID"], s2["Sample ID"]
        n1, n2 = int(tot.get(t1, 0)), int(tot.get(t2, 0))
        m1, m2 = int(mes.get(t1, 0)), int(mes.get(t2, 0))
        if n1 == 0 or n2 == 0:
            dropped.append((pid, "no malignant cells"))
            continue
        rows.append(dict(patient=pid, T1=t1, T2=t2, tp_total=n1, r1_total=n2,
                         tp_MES=m1, r1_MES=m2, tp_frac=m1 / n1, r1_frac=m2 / n2))
    return pd.DataFrame(rows), dropped


def median_ci(a, b, reps=5000):
    diff = np.median(a) - np.median(b)
    bs = [np.median(rng.choice(a, len(a))) - np.median(rng.choice(b, len(b)))
          for _ in range(reps)]
    return diff, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def distortion(counts, row):
    c1 = counts[counts.ID == row.T1].set_index("State").n
    c2 = counts[counts.ID == row.T2].set_index("State").n
    st = sorted(set(c1.index) | set(c2.index))
    t = c1.reindex(st).fillna(0).to_numpy(float)
    u = c2.reindex(st).fillna(0).to_numpy(float)
    i = st.index("MES")
    w1, w2 = t / t.sum(), u / u.sum()
    keep = (t >= C.MIN_CELLS) & (u >= C.MIN_CELLS)
    wt, wu = w1 * keep, w2 * keep
    if wt.sum() == 0 or wu.sum() == 0:
        return None
    wt, wu = wt / wt.sum(), wu / wu.sum()
    return dict(patient=row.patient, dw_all=w2[i] - w1[i], dw_renorm=wu[i] - wt[i],
                states_total=len(st), states_kept=int(keep.sum()),
                weight_dropped=max(w1[~keep].sum(), w2[~keep].sum()))


def main():
    cl = load_clinical()
    counts = pd.read_csv(C.CARE_WT_COUNTS)
    pairs, dropped = build_pairs(cl, counts)
    pairs.to_csv(os.path.join(C.OUT, "care_wt_pairs.csv"), index=False)
    print(f"strict pairs: {len(pairs)}   excluded at pairing: {len(dropped)}")
    for p, why in dropped:
        print(f"   {p}: {why}")

    kept = [int(((pairs.tp_MES >= t) & (pairs.r1_MES >= t)).sum()) for t in C.THRESHOLDS]
    pd.DataFrame({"min_cells": C.THRESHOLDS, "patients": kept}).to_csv(
        os.path.join(C.OUT, "care_wt_thresholds.csv"), index=False)
    print("thresholds:", dict(zip(C.THRESHOLDS, kept)))

    pairs["retained"] = (pairs.tp_MES >= 20) & (pairs.r1_MES >= 20)
    both = int(pairs.retained.sum())
    tp_only = int(((pairs.tp_MES < 20) & (pairs.r1_MES >= 20)).sum())
    r1_only = int(((pairs.tp_MES >= 20) & (pairs.r1_MES < 20)).sum())
    neither = int(((~pairs.retained) & (pairs.tp_MES < 20) & (pairs.r1_MES < 20)).sum())
    p_mcnemar = binomtest(r1_only, r1_only + tp_only).pvalue
    print(f"four-way: both {both}, TP only {tp_only}, R1 only {r1_only}, both fail {neither}")
    print(f"exact McNemar P = {p_mcnemar:.4f}")

    rows = []
    for col in ("tp_frac", "tp_total", "tp_MES", "r1_frac"):
        a = pairs.loc[pairs.retained, col].dropna().to_numpy()
        b = pairs.loc[~pairs.retained, col].dropna().to_numpy()
        d, lo, hi = median_ci(a, b)
        rows.append(dict(variable=col, retained=np.median(a), excluded=np.median(b),
                         difference=d, ci_low=lo, ci_high=hi))
    pd.DataFrame(rows).to_csv(os.path.join(C.OUT, "care_wt_between_group.csv"), index=False)
    for r in rows:
        print(f"{r['variable']:9s} retained {r['retained']:8.3f} excluded {r['excluded']:8.3f} "
              f"diff {r['difference']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]")

    dist = [d for d in (distortion(counts, r) for r in pairs[pairs.retained].itertuples())
            if d is not None]
    D = pd.DataFrame(dist)
    D["abs_diff"] = (D.dw_renorm - D.dw_all).abs()
    D.to_csv(os.path.join(C.OUT, "care_wt_distortion.csv"), index=False)
    print(f"distortion: n={len(D)} median|diff|={D.abs_diff.median():.4f} "
          f"mean={D.abs_diff.mean():.4f} max={D.abs_diff.max():.3f} "
          f"r={D.dw_all.corr(D.dw_renorm):.3f} "
          f"below 0.02={int((D.abs_diff < 0.02).sum())}")


if __name__ == "__main__":
    main()
