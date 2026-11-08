"""Ten-state eligibility audit on the same 52 paired patients.

For every annotated malignant-cell state, applies the same minimum-cell rule at both
endpoints, records which endpoint fails, builds the patient x state eligibility matrix,
and compares the observed overlap between state-specific eligible sets with the overlap
expected if eligibility were allocated independently at the observed margins.

Inputs
  data/care_wt_state_counts.csv
  outputs/care_wt_pairs.csv

Outputs (outputs/)
  states10_matrix.csv        patient x state binary eligibility
  states10_summary.csv       per-state four-way counts at each threshold
  states10_patients.csv      eligible-state count per patient
  states10_jaccard.csv       pairwise overlap, observed and expected
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

THRESHOLDS = [10, 20, 50]


def main():
    counts = pd.read_csv(C.CARE_WT_COUNTS)
    pairs = pd.read_csv(os.path.join(C.OUT, "care_wt_pairs.csv"))
    wide = counts.pivot_table(index="patient", columns=["tp", "State"], values="n",
                              aggfunc="sum").fillna(0)
    states = sorted(wide.columns.get_level_values(1).unique())

    for thr in THRESHOLDS:
        ok_t = wide.xs("T1", axis=1, level=0) >= thr
        ok_u = wide.xs("T2", axis=1, level=0) >= thr
        rows = []
        for st in states:
            t = ok_t[st].reindex(pairs.patient).fillna(False).to_numpy()
            u = ok_u[st].reindex(pairs.patient).fillna(False).to_numpy()
            rows.append(dict(
                threshold=thr, state=st, n_pairs=len(t),
                both=int((t & u).sum()),
                fails_primary_only=int((~t & u).sum()),
                fails_recurrence_only=int((t & ~u).sum()),
                both_fail=int((~t & ~u).sum()),
                retained=round(float((t & u).mean()), 3)))
        S = pd.DataFrame(rows).sort_values("both", ascending=False)
        S.to_csv(os.path.join(C.OUT, f"states10_summary_{thr}.csv"), index=False)
        if thr == C.MIN_CELLS:
            S.to_csv(os.path.join(C.OUT, "states10_summary.csv"), index=False)
            M = pd.DataFrame({st: (ok_t[st].reindex(pairs.patient).fillna(False).to_numpy()
                                  & ok_u[st].reindex(pairs.patient).fillna(False).to_numpy())
                              for st in states}, index=pairs.patient)
            M.to_csv(os.path.join(C.OUT, "states10_matrix.csv"))
        print(f"\nthreshold {thr}")
        print(S[["state", "both", "fails_primary_only", "fails_recurrence_only",
                 "both_fail", "retained"]].to_string(index=False))

    # patient-level summary at the primary threshold
    M = pd.read_csv(os.path.join(C.OUT, "states10_matrix.csv"), index_col=0)
    n_states = M.sum(axis=1)
    pd.DataFrame({"patient": n_states.index, "eligible_states": n_states.to_numpy()}).to_csv(
        os.path.join(C.OUT, "states10_patients.csv"), index=False)
    print(f"\npatients: {len(M)}   median eligible states {n_states.median():.0f}   "
          f"range {n_states.min()}-{n_states.max()}")
    print(f"zero eligible states: {int((n_states == 0).sum())}   "
          f"all ten: {int((n_states == 10).sum())}")

    # pairwise overlap, observed vs independent allocation at the same margins
    k = M.sum(axis=0)
    rows = []
    for i, a in enumerate(states):
        for b in states[i + 1:]:
            inter = int((M[a] & M[b]).sum())
            union = int((M[a] | M[b]).sum())
            exp_inter = k[a] * k[b] / len(M)
            exp_union = k[a] + k[b] - exp_inter
            rows.append(dict(state_a=a, state_b=b, n_a=int(k[a]), n_b=int(k[b]),
                             observed_intersection=inter, observed_jaccard=inter / union,
                             expected_intersection=exp_inter,
                             expected_jaccard=exp_inter / exp_union))
    J = pd.DataFrame(rows)
    J.to_csv(os.path.join(C.OUT, "states10_jaccard.csv"), index=False)
    print(f"\nmean Jaccard observed {J.observed_jaccard.mean():.3f}   "
          f"expected under independent allocation {J.expected_jaccard.mean():.3f}")
    p = k / len(M)
    print(f"expected patients with no eligible state {len(M) * np.prod(1 - p):.2f}   "
          f"observed {int((n_states == 0).sum())}")
    print(f"lowest observed overlap: "
          f"{J.nsmallest(1, 'observed_jaccard')[['state_a', 'state_b', 'observed_jaccard']].to_dict('records')}")


if __name__ == "__main__":
    main()
