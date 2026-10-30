import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

"""
P0-1 (rewritten): same estimand, per-gene metrics, fixed delta across replicates.

Two estimands
  T_CONTRIB(g) = mean_p(wbar_MES,p) * delta_g     within-state CONTRIBUTION to dA
  T_EFFECT(g)  = delta_g                          within-state EFFECT on MES cells

Estimators
  E0 naive dA                      -> T_CONTRIB + composition term
  E1 within component              -> T_CONTRIB
  E2 mean dx_MES (complete case)   -> T_EFFECT
  E3 ANCOVA intercept              -> T_CONTRIB

Metrics are computed per gene across replicates, then summarised within strata
(delta = 0 vs delta != 0). delta is FIXED across replicates.
"""
import json
import os

import numpy as np
import pandas as pd

RNG = np.random.default_rng(C.SEED)
OUT = C.OUT
NPAT, NGENE, R = 19, 2000, 200
THR = 20


def make_delta(sd):
    d = RNG.normal(0, sd, NGENE) if sd > 0 else np.zeros(NGENE)
    if sd > 0:
        d[RNG.random(NGENE) < 0.8] = 0.0
    return d


def run(name, offset=0.0, dw_sd=0.0, delta_sd=0.5, single=False, filt=False,
        rare=False):
    ns = 1 if single else 6
    delta = make_delta(delta_sd)
    nz = delta != 0
    est = {k: np.full((R, NGENE), np.nan) for k in ("E0", "E1", "E2", "E3")}
    wbar_used = np.zeros(R)
    kept_frac = np.zeros(R)

    for r in range(R):
        E0, E1, E2, dwv, wb = [], [], [], [], []
        kept = 0
        for _p in range(NPAT):
            alpha = np.r_[0.5, np.full(ns - 1, 2.0)] if rare else np.full(ns, 1.5)
            w1 = RNG.dirichlet(alpha)
            d = np.zeros(ns)
            if ns > 1:
                d[0] = RNG.normal(0, dw_sd)
            w2 = np.clip(w1 + d, 1e-3, None); w2 /= w2.sum()
            mu = RNG.normal(0, 1.0, (ns, NGENE)); mu[0] += offset
            x1 = mu + RNG.normal(0, 0.3, (ns, NGENE))
            x2 = x1 + RNG.normal(0, 0.3, (ns, NGENE)); x2[0] += delta

            keep = np.ones(ns, bool)
            if filt:
                c1 = RNG.lognormal(4.5, 0.9, ns); c2 = RNG.lognormal(4.5, 0.9, ns)
                keep = (c1 >= THR) & (c2 >= THR)
                w1f = np.where(keep, w1, 0.0); w2f = np.where(keep, w2, 0.0)
                if w1f.sum() == 0 or w2f.sum() == 0:
                    continue
                w1, w2 = w1f / w1f.sum(), w2f / w2f.sum()
            kept += int(keep[0])

            wbar = (w1 + w2) / 2
            E0.append(w2 @ x2 - w1 @ x1)
            E1.append(wbar @ (x2 - x1))
            E2.append(x2[0] - x1[0] if keep[0] else np.full(NGENE, np.nan))
            dwv.append(w2[0] - w1[0]); wb.append(wbar[0])

        E0, E1, E2 = map(np.array, (E0, E1, E2))
        dwv = np.array(dwv)
        b = dwv - dwv.mean(); sxx = (b ** 2).sum()
        if sxx < 1e-12:
            E3 = np.tile(np.nanmean(E0, axis=0), (len(E0), 1))
        else:
            slope = (b[:, None] * E0).sum(0) / sxx
            E3 = E0 - (b[:, None] * slope[None, :])
        est["E0"][r] = np.nanmean(E0, axis=0)
        est["E1"][r] = np.nanmean(E1, axis=0)
        est["E2"][r] = np.nanmean(E2, axis=0)
        est["E3"][r] = np.nanmean(E3, axis=0)
        wbar_used[r] = np.mean(wb)
        kept_frac[r] = kept / NPAT

    T = wbar_used.mean() * delta
    rows = {"scenario": name, "mean_wbar": float(wbar_used.mean()),
            "frac_kept_mean": float(kept_frac.mean()),
            "frac_delta_nonzero": float(nz.mean())}
    for k in ("E0", "E1", "E2", "E3"):
        v = est[k]
        target = T if k in ("E0", "E1", "E3") else delta
        mean_est = np.nanmean(v, axis=0)
        rows[f"{k}_bias"] = float(np.nanmean(mean_est - target))
        rows[f"{k}_rmse"] = float(
            np.nanmean(np.sqrt(np.nanmean((v - target) ** 2, axis=0))))
        if nz.any():
            rows[f"{k}_signerr"] = float(np.nanmean(
                (np.sign(v[:, nz]) != np.sign(target[nz])).astype(float)))
        z = ~nz
        if z.any():
            rows[f"{k}_null_sd"] = float(np.nanmean(np.nanstd(v[:, z], axis=0)))
            rows[f"{k}_null_big"] = float(np.nanmean(np.abs(v[:, z]) > 0.25))
    return rows


SCEN = [
    ("S1 no composition change", dict(dw_sd=0.0, delta_sd=0.5)),
    ("S2 composition only", dict(offset=1.0, dw_sd=0.10, delta_sd=0.0)),
    ("S3 both +offset", dict(offset=1.0, dw_sd=0.10, delta_sd=0.5)),
    ("S4 both -offset", dict(offset=-1.0, dw_sd=0.10, delta_sd=0.5)),
    ("S5 single state", dict(dw_sd=0.0, delta_sd=0.5, single=True)),
    ("S6 filter on", dict(offset=1.0, dw_sd=0.10, delta_sd=0.5, filt=True)),
    ("S7 filter, rare MES", dict(offset=1.0, dw_sd=0.10, delta_sd=0.5, filt=True,
                                 rare=True)),
]

rows = [run(n, **kw) for n, kw in SCEN]
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "P0_1_corrected_benchmark.tsv"), sep="\t", index=False)

show = ["scenario", "mean_wbar", "frac_kept_mean",
        "E0_bias", "E1_bias", "E2_bias", "E3_bias",
        "E1_rmse", "E2_rmse", "E3_rmse",
        "E1_signerr", "E2_signerr", "E3_signerr"]
print(df[show].round(3).to_string(index=False))
json.dump(rows, open(os.path.join(OUT, "P0_1_summary.json"), "w"),
          indent=2, ensure_ascii=False)

