"""Recovery-adjusted calibration (repo version, config paths, 52 pairs only, outputs saved).

Keeps each patient's real malignant-cell totals, draws state counts multinomially from the
cohort-wide composition of the same 52 paired specimens, and applies the 20-cell rule.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

MIN = C.MIN_CELLS
REPS = 400


def main():
    counts = pd.read_csv(C.CARE_WT_COUNTS)
    pairs = pd.read_csv(os.path.join(C.OUT, "care_wt_pairs.csv"))
    keep = set(pairs.patient)
    counts = counts[counts.patient.isin(keep) & counts.tp.isin(["T1", "T2"])]

    wide = counts.pivot_table(index="patient", columns=["tp", "State"], values="n",
                              aggfunc="sum").fillna(0)
    states = sorted(wide.columns.get_level_values(1).unique())
    t1 = wide.xs("T1", axis=1, level=0)[states].reindex(pairs.patient).fillna(0)
    t2 = wide.xs("T2", axis=1, level=0)[states].reindex(pairs.patient).fillna(0)

    p1 = t1.sum() / t1.sum().sum()
    p2 = t2.sum() / t2.sum().sum()
    N1 = pairs.set_index("patient").loc[t1.index, "tp_total"].to_numpy()
    N2 = pairs.set_index("patient").loc[t1.index, "r1_total"].to_numpy()
    minN = np.minimum(N1, N2)

    obs_ok = (t1.to_numpy() >= MIN) & (t2.to_numpy() >= MIN)
    obs_rho = spearmanr(minN, obs_ok.sum(axis=1)).statistic
    js = [((obs_ok[:, i] & obs_ok[:, j]).sum() / (obs_ok[:, i] | obs_ok[:, j]).sum())
          for i in range(len(states)) for j in range(i + 1, len(states))]
    obs = dict(rho=float(obs_rho), zero=int((obs_ok.sum(axis=1) == 0).sum()),
               jaccard=float(np.mean(js)), per_state=obs_ok.sum(axis=0).tolist())

    rng = np.random.default_rng(C.SEED)
    draws = []
    for _ in range(REPS):
        c1 = rng.multinomial(N1.astype(int), p1.to_numpy())
        c2 = rng.multinomial(N2.astype(int), p2.to_numpy())
        ok = (c1 >= MIN) & (c2 >= MIN)
        k = ok.sum(axis=1)
        draws.append(dict(rho=float(spearmanr(minN, k).statistic),
                          zero=int((k == 0).sum()),
                          jaccard=float(np.mean([((ok[:, i] & ok[:, j]).sum()
                                                  / (ok[:, i] | ok[:, j]).sum())
                                                 for i in range(len(states))
                                                 for j in range(i + 1, len(states))])),
                          per_state=ok.sum(axis=0).tolist()))

    def summarise(key):
        v = np.array([d[key] for d in draws], dtype=float)
        return dict(mean=float(v.mean()), lo=float(np.percentile(v, 2.5)),
                    hi=float(np.percentile(v, 97.5)))

    out = dict(observed=obs, reference={k: summarise(k) for k in ("rho", "zero", "jaccard")},
               draws=draws, reps=REPS, states=states,
               note="reference distribution, not a confidence interval for the observed effect")
    pd.DataFrame([{k: v for k, v in d.items() if k != "per_state"} for d in draws]).to_csv(
        os.path.join(C.OUT, "calibration_recovery_draws.csv"), index=False)
    with open(os.path.join(C.OUT, "calibration_recovery.json"), "w") as fh:
        json.dump(out, fh, indent=1)

    print(f"observed rho {obs['rho']:.3f}  zero {obs['zero']}  jaccard {obs['jaccard']:.3f}")
    for k in ("rho", "zero", "jaccard"):
        s = out["reference"][k]
        print(f"reference {k:8s} {s['mean']:.3f} [{s['lo']:.3f}, {s['hi']:.3f}]")


if __name__ == "__main__":
    main()
