"""Longitudinal MES-fraction change: full paired cohort versus the eligible subset.

Compares the within-patient change in MES-like fraction between all strictly
paired patients, the patients eligible under the 20-cell rule, and the excluded
patients. Because the eligible set is a subset of the paired set, the difference
between the two medians is evaluated with a nested bootstrap: the 52 patients are
resampled with replacement and the 20-cell rule is re-applied in every draw.

Inputs
  outputs/care_wt_pairs.csv    written by scripts/01_care_wt_eligibility.py

Outputs (outputs/)
  care_wt_longitudinal_dmes.csv
  care_wt_longitudinal_dmes.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

rng = np.random.default_rng(C.SEED)
B = 20000


def ci(x, stat=np.median):
    idx = rng.integers(0, len(x), (B, len(x)))
    return np.percentile(stat(x[idx], axis=1), [2.5, 97.5])


def main():
    P = pd.read_csv(os.path.join(C.OUT, "care_wt_pairs.csv"))
    P["dMES"] = P.r1_frac - P.tp_frac
    P["eligible"] = (P.tp_MES >= C.MIN_CELLS) & (P.r1_MES >= C.MIN_CELLS)
    P["min_cells"] = P[["tp_total", "r1_total"]].min(axis=1)

    out = {}
    for name, sub in (("all", P), ("eligible", P[P.eligible]), ("excluded", P[~P.eligible])):
        x = sub.dMES.to_numpy()
        lo, hi = ci(x)
        out[name] = dict(n=int(len(x)), median=float(np.median(x)),
                         ci_low=float(lo), ci_high=float(hi),
                         min_cells_median=float(sub.min_cells.median()))
        print(f"{name:9s} n={len(x):2d} median {np.median(x):+.4f} [{lo:+.4f}, {hi:+.4f}] "
              f"median min cells {sub.min_cells.median():.0f}")

    # nested bootstrap of (eligible - all)
    n = len(P)
    boot = P.iloc[rng.integers(0, n, (B, n)).ravel()].reset_index(drop=True)
    boot["b"] = np.repeat(np.arange(B), n)
    boot["el"] = (boot.tp_MES >= C.MIN_CELLS) & (boot.r1_MES >= C.MIN_CELLS)
    diff = (boot[boot.el].groupby("b").dMES.median()
            - boot.groupby("b").dMES.median()).dropna()

    point = float(P[P.eligible].dMES.median() - P.dMES.median())
    lo, hi = np.percentile(diff, [2.5, 97.5])
    out["difference_eligible_minus_all"] = dict(
        point=point, ci_low=float(lo), ci_high=float(hi),
        method="nested bootstrap: resample 52 patients, re-apply the 20-cell rule each draw",
        draws=int(len(diff)))
    print(f"\neligible - all: {point:+.4f} [{lo:+.4f}, {hi:+.4f}] (nested, B={B})")

    P.to_csv(os.path.join(C.OUT, "care_wt_longitudinal_dmes.csv"), index=False)
    with open(os.path.join(C.OUT, "care_wt_longitudinal_dmes.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
