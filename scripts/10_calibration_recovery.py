"""Recovery-adjusted calibration.

Keeps each patient's real malignant-cell totals at both endpoints and the cohort-wide
state proportions, resamples the state counts multinomially, and applies the same
20-cell rule. Tests how much of the observed eligibility structure follows from cell
recovery and shared composition alone.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MIN = 20
S = pd.read_csv(r"C:\work\repo_glioma_filter\data\care_wt_state_counts.csv")
P = pd.read_csv(r"C:\work\paper_v3\audit_pairs52.csv")

wide = S.pivot_table(index="patient", columns=["tp", "State"], values="n", aggfunc="sum").fillna(0)
states = sorted(wide.columns.get_level_values(1).unique())

t1 = wide.xs("T1", axis=1, level=0)[states]
t2 = wide.xs("T2", axis=1, level=0)[states]
pooled1 = t1.sum() / t1.sum().sum()
pooled2 = t2.sum() / t2.sum().sum()

N1 = P.set_index("patient").tp_total.reindex(P.patient).to_numpy()
N2 = P.set_index("patient").r1_total.reindex(P.patient).to_numpy()
minN = np.minimum(N1, N2)

rng = np.random.default_rng(20261029)
reps = 400
null = []
for _ in range(reps):
    c1 = rng.multinomial(N1.astype(int), pooled1.to_numpy())
    c2 = rng.multinomial(N2.astype(int), pooled2.to_numpy())
    ok = (c1 >= MIN) & (c2 >= MIN)
    n_states = ok.sum(axis=1)
    rho = spearmanr(minN, n_states).statistic
    js = []
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            u = (ok[:, i] | ok[:, j]).sum()
            js.append((ok[:, i] & ok[:, j]).sum() / u if u else 0.0)
    null.append(dict(rho=rho, zero=int((n_states == 0).sum()), jaccard=np.mean(js),
                     per_state=ok.sum(axis=0)))

N = pd.DataFrame([{k: v for k, v in d.items() if k != "per_state"} for d in null])
obs_zero = 6
obs_rho = 0.900
obs_jac = 0.431

print(f"calibration, {reps} draws")
print(f"  rho      observed {obs_rho:.3f}   calibration {N.rho.mean():.3f} "
      f"[{N.rho.quantile(.025):.3f}, {N.rho.quantile(.975):.3f}]")
print(f"  zero     observed {obs_zero}      calibration {N.zero.mean():.1f} "
      f"[{N.zero.quantile(.025):.0f}, {N.zero.quantile(.975):.0f}]")
print(f"  Jaccard  observed {obs_jac:.3f}   calibration {N.jaccard.mean():.3f} "
      f"[{N.jaccard.quantile(.025):.3f}, {N.jaccard.quantile(.975):.3f}]")

ps = np.array([d["per_state"] for d in null])
print("\nper-state eligible pairs: observed vs calibration mean [95%]")
obs = (t1.reindex(P.patient).to_numpy() >= MIN) & (t2.reindex(P.patient).to_numpy() >= MIN)
for k, st in enumerate(states):
    print(f"  {st:11s} {int(obs[:, k].sum()):3d}   {ps[:, k].mean():6.1f} "
          f"[{np.percentile(ps[:, k], 2.5):.1f}, {np.percentile(ps[:, k], 97.5):.1f}]")
