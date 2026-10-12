"""Panel synthétique avec alphas PLANTÉS connus — pour valider que le pipeline
retrouve ce qui existe (et ne trouve rien quand il n'y a rien : alpha_scale=0).

Alphas plantés :
 - erreur de prix transitoire (AR(1), demi-vie ~4 barres) : retour à la moyenne
   idiosyncratique sur plusieurs barres, chocs 2.5x plus gros en régime de stress
   (=> la régression conditionnelle doit le voir)
 - dérive lente persistante (OU) => momentum
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl


def make_synthetic(n_symbols: int = 300, n_bars: int = 2500, seed: int = 0,
                   alpha_scale: float = 1.0) -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(seed)
    T, N = n_bars, n_symbols
    # marché à régimes de vol (chaîne de Markov)
    state = np.zeros(T, dtype=int)
    for t in range(1, T):
        stay = 0.985 if state[t - 1] == 0 else 0.95
        state[t] = state[t - 1] if rng.random() < stay else 1 - state[t - 1]
    mvol = np.where(state == 1, 0.022, 0.008)
    mkt = rng.normal(0.0003, 1, T) * mvol

    beta = rng.uniform(0.5, 1.5, N)
    ivol = rng.uniform(0.01, 0.03, N)
    mu = np.zeros(N)
    z = np.zeros(N)                     # erreur de prix transitoire
    R = np.empty((T, N))
    U = np.empty((T, N))
    for t in range(T):
        mu = 0.985 * mu + alpha_scale * rng.normal(0, 0.00012, N)   # dérive persistante
        eps = rng.standard_t(5, N) / np.sqrt(5 / 3) * ivol * (1 + 0.6 * state[t])  # queues épaisses
        z_new = 0.85 * z + alpha_scale * rng.normal(0, 0.25, N) * ivol * (1 + 1.5 * state[t])
        u = eps + (z_new - z)
        z = z_new
        R[t] = beta * mkt[t] + mu + u
        U[t] = u
    logp = np.log(50) + np.cumsum(R, axis=0) + rng.normal(0, 0.5, N)
    close = np.exp(logp)
    prev = np.vstack([close[:1], close[:-1]])
    opn = prev * np.exp(rng.normal(0, 0.002, (T, N)))
    hi = np.maximum(opn, close) * np.exp(np.abs(rng.normal(0, 0.004, (T, N))))
    lo = np.minimum(opn, close) * np.exp(-np.abs(rng.normal(0, 0.004, (T, N))))
    base_v = rng.lognormal(13, 1, N)
    vol = base_v * np.exp(1.5 * np.abs(U) / ivol / (1 + state[:, None]) * 0.3 + rng.normal(0, 0.3, (T, N)))

    ts = [dt.datetime(2015, 1, 1) + dt.timedelta(days=i) for i in range(T)]
    syms = [f"S{i:04d}" for i in range(N)]
    bars = pl.DataFrame({
        "ts": np.repeat(np.array(ts, dtype="datetime64[us]"), N),
        "symbol": np.tile(syms, T),
        "open": opn.ravel(), "high": hi.ravel(), "low": lo.ravel(),
        "close": close.ravel(), "volume": vol.ravel(),
    })
    vix = 100 * mvol * np.sqrt(252) * np.exp(rng.normal(0, 0.08, T))
    market = pl.DataFrame({"ts": np.array(ts, dtype="datetime64[us]"), "mkt_ret": mkt, "vix": vix})
    return bars, market
