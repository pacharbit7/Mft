"""Régressions conditionnelles.

1) RegimeRidge  — coefficients par régime, rétrécis vers le modèle poolé :
       b_r = argmin  1/n_r Σ_{i∈r} w(y - Xb)^2 + lam_r |b - b_pool|^2
   Solution fermée : b_r = (G_r + lam_r I)^-1 (c_r + lam_r b_pool).
   lam_r -> inf : pas de conditionnement ; lam_r -> 0 : régressions séparées.
   C'est l'estimateur "hiérarchique" : un régime peu peuplé (stress) emprunte
   de la force au pool au lieu de sur-ajuster.

2) InteractionModel — coefficients qui varient continûment avec z (vol, VIX…) :
       y = X b + (X ⊙ z) g      =>  b(z) = b + g z
   On l'entraîne avec n'importe quel modèle linéaire (ridge/lasso).
"""
from __future__ import annotations

import numpy as np

from .linear import Ridge, moments


class RegimeRidge:
    def __init__(self, lam_pool: float = 1.0, lam_regime: float = 5.0, min_obs: int = 500):
        self.lam_pool, self.lam_regime, self.min_obs = lam_pool, lam_regime, min_obs

    def fit(self, X, y, regime, w=None):
        w = np.ones(len(y)) if w is None else w
        by = {r: (moments(X[regime == r], y[regime == r], w[regime == r]), int((regime == r).sum()))
              for r in np.unique(regime) if r >= 0}          # -1 = régime inconnu -> pool
        return self.fit_moments(moments(X, y, w), by)

    def fit_moments(self, pool_mom, by_regime: dict):
        """pool_mom = (xm, ym, G, c) ; by_regime = {r: ((xm, ym, G, c), n_obs)}."""
        pool = Ridge(self.lam_pool).fit_gram(*pool_mom)
        self.pool_ = pool
        # échelle commune = celle du pool (pénalité comparable entre régimes)
        sd = np.sqrt(np.clip(np.diag(pool_mom[2]), 1e-12, None))
        b_pool_std = pool.coef_ * sd
        self.coefs_, self.intercepts_ = {}, {}
        for r, ((xm, ym, Gr, cr), n) in by_regime.items():
            if n < self.min_obs:
                continue
            Gs = Gr / np.outer(sd, sd)
            cs = cr / sd
            b = np.linalg.solve(Gs + self.lam_regime * np.eye(len(cs)), cs + self.lam_regime * b_pool_std) / sd
            self.coefs_[r] = b
            self.intercepts_[r] = float(ym - xm @ b)
        return self

    def predict(self, X, regime):
        out = self.pool_.predict(X)
        for r, b in self.coefs_.items():
            m = regime == r
            out[m] = X[m] @ b + self.intercepts_[r]
        return out


def add_interactions(X: np.ndarray, z: np.ndarray, names: list[str] | None = None,
                     zname: str = "z"):
    """[X, X*z]. z doit être connu en t (ex : vol_z marché) et ~standardisé."""
    z = np.nan_to_num(z, nan=0.0)
    Xi = np.hstack([X, X * z[:, None]])
    if names is None:
        return Xi
    return Xi, names + [f"{n}*{zname}" for n in names]


class InteractionModel:
    """Enveloppe : base_model entraîné sur [X, X*z]."""

    def __init__(self, base_model):
        self.base = base_model

    def fit(self, X, y, z, w=None):
        return self.fit_gram(*moments(add_interactions(X, z), y, w))

    def fit_gram(self, xm, ym, G, c):
        self.base.fit_gram(xm, ym, G, c)
        p = len(c) // 2
        self.coef_main_ = self.base.coef_[:p]
        self.coef_inter_ = self.base.coef_[p:]
        return self

    def predict(self, X, z):
        return self.base.predict(add_interactions(X, z))
