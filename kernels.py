"""Kernels Numba.

Convention : les tableaux sont triés par groupe puis par temps ; `offsets`
(longueur n_groupes + 1) délimite chaque groupe contigu.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from numba import njit, prange


def offsets_from(df: pl.DataFrame, key: str) -> np.ndarray:
    """Offsets des groupes contigus de `key` (le df doit déjà être trié par `key`)."""
    ids = df.select(pl.col(key).rle_id()).to_series().to_numpy()
    if ids.size == 0:
        return np.zeros(1, dtype=np.int64)
    cuts = np.flatnonzero(np.diff(ids)) + 1
    return np.concatenate(([0], cuts, [ids.size])).astype(np.int64)


@njit(parallel=True, cache=True)
def rolling_ols_resid(y, x, offsets, window, min_obs):
    """Régression glissante y = a + b*x par groupe (sommes glissantes, O(n)).

    Retourne (beta, résidu). Le résidu en t n'utilise que des données <= t.
    """
    n = y.shape[0]
    beta = np.full(n, np.nan)
    resid = np.full(n, np.nan)
    for g in prange(offsets.shape[0] - 1):
        s, e = offsets[g], offsets[g + 1]
        sx = sy = sxx = sxy = 0.0
        cnt = 0
        for i in range(s, e):
            xi, yi = x[i], y[i]
            if not (np.isnan(xi) or np.isnan(yi)):
                sx += xi; sy += yi; sxx += xi * xi; sxy += xi * yi; cnt += 1
            j = i - window
            if j >= s:
                xj, yj = x[j], y[j]
                if not (np.isnan(xj) or np.isnan(yj)):
                    sx -= xj; sy -= yj; sxx -= xj * xj; sxy -= xj * yj; cnt -= 1
            if cnt >= min_obs:
                mx, my = sx / cnt, sy / cnt
                vx = sxx / cnt - mx * mx
                if vx > 1e-14:
                    b = (sxy / cnt - mx * my) / vx
                    beta[i] = b
                    if not (np.isnan(xi) or np.isnan(yi)):
                        resid[i] = yi - (my - b * mx) - b * xi
    return beta, resid


@njit(cache=True)
def _rank(a):
    order = np.argsort(a)
    r = np.empty(a.size)
    for i in range(a.size):
        r[order[i]] = i
    return r


@njit(parallel=True, cache=True)
def grouped_ic(a, b, offsets, use_rank, min_n):
    """Corrélation (Pearson ou Spearman) entre a et b dans chaque groupe (= chaque date)."""
    ng = offsets.shape[0] - 1
    out = np.full(ng, np.nan)
    for g in prange(ng):
        s, e = offsets[g], offsets[g + 1]
        cnt = 0
        for i in range(s, e):
            if not (np.isnan(a[i]) or np.isnan(b[i])):
                cnt += 1
        if cnt < min_n:
            continue
        x = np.empty(cnt)
        y = np.empty(cnt)
        k = 0
        for i in range(s, e):
            if not (np.isnan(a[i]) or np.isnan(b[i])):
                x[k] = a[i]; y[k] = b[i]; k += 1
        if use_rank:
            x = _rank(x)
            y = _rank(y)
        mx, my = x.mean(), y.mean()
        sxy = sxx = syy = 0.0
        for i in range(cnt):
            dx, dy = x[i] - mx, y[i] - my
            sxy += dx * dy; sxx += dx * dx; syy += dy * dy
        if sxx > 0 and syy > 0:
            out[g] = sxy / np.sqrt(sxx * syy)
    return out


@njit(cache=True)
def enet_cd_gram(G, c, lam, alpha, beta0, positive, max_iter, tol):
    """Elastic net par descente de coordonnées sur la matrice de Gram.

    min 1/2 b'Gb - c'b + lam*(alpha*|b|_1 + (1-alpha)/2*|b|^2)
    équivalent à 1/(2n)||y-Xb||^2 + pénalité, avec G = X'X/n, c = X'y/n.
    Coût O(p^2) par passe, indépendant de n : très rapide quand n >> p.
    """
    p = G.shape[0]
    beta = beta0.copy()
    Gb = G @ beta
    l1 = lam * alpha
    l2 = lam * (1.0 - alpha)
    for _ in range(max_iter):
        max_d = 0.0
        for j in range(p):
            bj = beta[j]
            rho = c[j] - Gb[j] + G[j, j] * bj
            den = G[j, j] + l2
            if rho > l1:
                nb = (rho - l1) / den
            elif rho < -l1 and not positive:
                nb = (rho + l1) / den
            else:
                nb = 0.0
            d = nb - bj
            if d != 0.0:
                for k in range(p):
                    Gb[k] += G[k, j] * d
                beta[j] = nb
                if abs(d) > max_d:
                    max_d = abs(d)
        if max_d < tol:
            break
    return beta


@njit(parallel=True, cache=True)
def date_moments(X, y, w, offsets):
    """Statistiques suffisantes PAR DATE : Σw, Σwx, Σwy, Σwxx', Σwxy, n.

    Ensuite, les moments de n'importe quel ensemble de dates = simple somme :
    un fit ridge/lasso coûte O(D·p²) au lieu de O(n·p²). C'est ce qui rend le
    walk-forward quasi instantané.
    """
    D = offsets.shape[0] - 1
    p = X.shape[1]
    sw = np.zeros(D); sy = np.zeros(D); cnt = np.zeros(D)
    sx = np.zeros((D, p)); sxy = np.zeros((D, p)); sxx = np.zeros((D, p, p))
    for d in prange(D):
        for i in range(offsets[d], offsets[d + 1]):
            wi = w[i]; yi = y[i]
            sw[d] += wi; sy[d] += wi * yi; cnt[d] += 1
            for a in range(p):
                xa = X[i, a] * wi
                sx[d, a] += xa
                sxy[d, a] += xa * yi
                for b in range(a, p):
                    sxx[d, a, b] += xa * X[i, b]
        for a in range(p):
            for b in range(a):
                sxx[d, a, b] = sxx[d, b, a]
    return sw, sx, sy, sxx, sxy, cnt
