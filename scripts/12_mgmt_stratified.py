"""MGMT promoter methylation: eligibility by group and robustness of the longitudinal difference.

Reproduces Table 2 and the group-matched random-retention reference reported in the manuscript.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

rng = np.random.default_rng(C.SEED)
B = 5000


def main():
    cl = pd.read_excel(C.CARE_WT_CLINICAL, sheet_name="TableS1", header=2)
    cl.columns = [str(c).strip() for c in cl.columns]
    cl["Patient ID"] = cl["Patient ID"].astype(str).str.strip()
    prim = cl[cl["Primary or recurrent"].astype(str).str.strip().str.lower()
              .str.startswith("primary")]
    status = (prim.groupby("Patient ID")["MGMT protomoter methylation status"]
              .agg(lambda s: s.dropna().iloc[0] if s.notna().any() else None))

    P = pd.read_csv(os.path.join(C.OUT, "care_wt_pairs.csv"))
    P["status"] = P.patient.map(status).astype(str).str.strip().str.lower()
    K = P[P.status.isin(["methylated", "unmethylated"])].copy()
    K["meth"] = K.status == "methylated"
    K["dMES"] = K.r1_frac - K.tp_frac
    ok = (K.tp_MES >= C.MIN_CELLS) & (K.r1_MES >= C.MIN_CELLS)

    tab = [[int((ok & K.meth).sum()), int((~ok & K.meth).sum())],
           [int((ok & ~K.meth).sum()), int((~ok & ~K.meth).sum())]]
    odds, pval = fisher_exact(tab)
    print(f"eligible by MGMT: methylated {tab[0][0]}/{sum(tab[0])}, "
          f"unmethylated {tab[1][0]}/{sum(tab[1])}; OR {odds:.2f}, Fisher P {pval:.3f}")

    rows = []
    for thr in (0, 10, 20, 50):
        s = K[(K.tp_MES >= thr) & (K.r1_MES >= thr)]
        a = s.loc[s.meth, "dMES"].to_numpy()
        b = s.loc[~s.meth, "dMES"].to_numpy()
        d = float(np.median(a) - np.median(b))
        ia = rng.integers(0, len(a), (B, len(a)))
        ib = rng.integers(0, len(b), (B, len(b)))
        dd = np.median(a[ia], axis=1) - np.median(b[ib], axis=1)
        rows.append(dict(rule=thr, n_meth=len(a), n_unmeth=len(b), theta=d,
                         lo=float(np.percentile(dd, 2.5)), hi=float(np.percentile(dd, 97.5))))
        print(f"  >= {thr:2d}: meth {len(a):2d}  unmeth {len(b):2d}  theta {d:+.4f} "
              f"[{rows[-1]['lo']:+.4f}, {rows[-1]['hi']:+.4f}]")

    keep_a, keep_b = int((ok & K.meth).sum()), int((ok & ~K.meth).sum())
    A, Bm = K.loc[K.meth, "dMES"].to_numpy(), K.loc[~K.meth, "dMES"].to_numpy()
    null = np.array([np.median(rng.choice(A, keep_a, replace=False))
                     - np.median(rng.choice(Bm, keep_b, replace=False)) for _ in range(B)])
    base, obs = rows[0]["theta"], rows[2]["theta"]
    shift = abs(obs - base)
    p_ref = float(np.mean(np.abs(null - base) >= shift))
    print(f"  random retention: keep {keep_a} of {len(A)} and {keep_b} of {len(Bm)}; "
          f"shift {shift:.4f}; 95th pct {np.percentile(np.abs(null - base), 95):.4f}; P {p_ref:.3f}")

    pd.DataFrame(rows).to_csv(os.path.join(C.OUT, "mgmt_theta_by_rule.csv"), index=False)
    json.dump(dict(table=tab, odds_ratio=odds, fisher_p=pval, rows=rows, shift=shift,
                   reference_p95=float(np.percentile(np.abs(null - base), 95)), p_ref=p_ref),
              open(os.path.join(C.OUT, "mgmt_module.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
