"""Métriques statistiquement honnêtes.

- IC par date (Spearman), puis t-stat Newey-West (les IC sont autocorrélés
  quand l'horizon chevauche : la t-stat naïve surestime la significativité).
- Deflated Sharpe Ratio : corrige le Sharpe du nombre d'essais (data-snooping),
  de la non-normalité (skew/kurtosis) et de la longueur de l'échantillon.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import polars as pl

from ..features.kernels import grouped_ic, offsets_from

_N = NormalDist()


def nw_tstat(x: np.ndarray, lags: int) -> float:
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return float("nan")
    e = x - x.mean()
    s = e @ e / n
    for l in range(1, lags + 1):
        s += 2 * (1 - l / (lags + 1)) * (e[l:] @ e[:-l]) / n
    return float(x.mean() / math.sqrt(max(s, 1e-20) / n))


def ic_series(df: pl.DataFrame, pred: str, target: str, rank: bool = True, min_n: int = 20) -> pl.DataFrame:
    d = df.select("ts", pred, target).sort("ts")
    off = offsets_from(d, "ts")
    ic = grouped_ic(d[pred].fill_null(np.nan).to_numpy().astype(float),
                    d[target].fill_null(np.nan).to_numpy().astype(float), off, rank, min_n)
    return pl.DataFrame({"ts": d["ts"].unique(maintain_order=True), "ic": ic})


def ic_summary(ic: np.ndarray, horizon: int = 1, bars_per_year: int = 252) -> dict:
    ic = ic[np.isfinite(ic)]
    return {
        "ic_mean": float(ic.mean()),
        "ic_std": float(ic.std()),
        "icir_ann": float(ic.mean() / ic.std() * math.sqrt(bars_per_year)),
        "t_nw": nw_tstat(ic, lags=max(horizon, 1) + 2),
        "hit": float((ic > 0).mean()),
        "n_dates": int(len(ic)),
    }


def ic_by_signal(df: pl.DataFrame, signals: list[str], target: str, horizon: int = 1) -> pl.DataFrame:
    rows = []
    for s in signals:
        ic = ic_series(df, s, target)["ic"].to_numpy()
        rows.append({"signal": s, **ic_summary(ic, horizon)})
    return pl.DataFrame(rows).sort("t_nw", descending=True, nulls_last=True)


def sharpe_stats(r: np.ndarray, bars_per_year: int = 252) -> dict:
    r = r[np.isfinite(r)]
    mu, sd = r.mean(), r.std(ddof=1)
    sr = mu / sd if sd > 0 else float("nan")
    eq = np.cumsum(r)
    dd = float((eq - np.maximum.accumulate(eq)).min())
    z = (r - mu) / sd
    return {
        "sharpe_ann": float(sr * math.sqrt(bars_per_year)),
        "ret_ann": float(mu * bars_per_year),
        "vol_ann": float(sd * math.sqrt(bars_per_year)),
        "max_dd": dd,
        "sr_bar": float(sr),
        "skew": float((z ** 3).mean()),
        "kurt": float((z ** 4).mean()),
        "n": int(len(r)),
    }


def deflated_sharpe(sr_bar: float, n: int, skew: float, kurt: float,
                    n_trials: int, var_trials_sr: float) -> float:
    """Proba que le vrai Sharpe > 0 compte tenu de n_trials essais (Bailey & López de Prado 2014).

    sr_bar et var_trials_sr sont en unités PAR BARRE (non annualisées).
    """
    g = 0.5772156649
    if n_trials > 1:
        sr0 = math.sqrt(max(var_trials_sr, 1e-12)) * (
            (1 - g) * _N.inv_cdf(1 - 1 / n_trials) + g * _N.inv_cdf(1 - 1 / (n_trials * math.e)))
    else:
        sr0 = 0.0
    den = math.sqrt(max(1 - skew * sr_bar + (kurt - 1) / 4 * sr_bar ** 2, 1e-12))
    return _N.cdf((sr_bar - sr0) * math.sqrt(n - 1) / den)


def ic_by_regime(df: pl.DataFrame, signals: list[str], target: str, regime_col: str = "regime",
                 horizon: int = 1) -> pl.DataFrame:
    """IC moyen et t-stat NW de chaque signal, conditionnellement au régime.

    Diagnostic de base AVANT toute régression conditionnelle : si l'IC ne change
    pas significativement entre régimes, conditionner n'ajoutera que du bruit.
    """
    rows = []
    for r in sorted(df[regime_col].drop_nulls().unique().to_list()):
        sub = df.filter(pl.col(regime_col) == r)
        for s in signals:
            ic = ic_series(sub, s, target)["ic"].to_numpy()
            st = ic_summary(ic, horizon)
            rows.append({"signal": s, "regime": r, "ic": st["ic_mean"], "t": st["t_nw"]})
    return (pl.DataFrame(rows).pivot(on="regime", index="signal", values=["ic", "t"]))
