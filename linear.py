"""Modèles linéaires régularisés, rapides (n >> p) et robustes.

Tout passe par les moments (G = X'WX/Σw, c = X'Wy/Σw) : une fois calculés,
le coût ne dépend plus de n. X est standardisé en interne pour que la
pénalité traite tous les signaux équitablement ; les coefs sont restitués
dans l'échelle d'origine.
"""
from __future__ import annotations

import numpy as np

from ..features.kernels import enet_cd_gram


def moments(X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None):
    """Moyennes pondérées, Gram centré et covariance X-y (BLAS, une seule copie)."""
    if w is None:
        sw = float(len(y))
        xm, ym = X.mean(0), y.mean()
        G = (X.T @ X) / sw - np.outer(xm, xm)
        c = (X.T @ y) / sw - xm * ym
        return xm, ym, G, c
    sw = w.sum()
    Xw = X * w[:, None]
    xm = Xw.sum(0) / sw
    ym = (w @ y) / sw
    G = (Xw.T @ X) / sw - np.outer(xm, xm)
    c = (Xw.T @ y) / sw - xm * ym
    return xm, ym, G, c


class MomentCache:
    """Moments par date précalculés (Numba) ; moments(dates) = somme sur ces dates.

    z (optionnel, une valeur PAR DATE) : moments de la régression augmentée [X, X*z]
    sans jamais matérialiser X*z.
    """

    def __init__(self, X, y, w, offsets):
        from ..features.kernels import date_moments
        w = np.ones(len(y)) if w is None else w
        self.sw, self.sx, self.sy, self.sxx, self.sxy, self.cnt = date_moments(
            np.ascontiguousarray(X), y, w, offsets)

    def n_rows(self, d) -> int:
        return int(self.cnt[d].sum())

    def moments(self, d: np.ndarray, z: np.ndarray | None = None):
        sw = self.sw[d].sum()
        if z is None:
            sx, sy = self.sx[d].sum(0), self.sy[d].sum()
            sxx, sxy = self.sxx[d].sum(0), self.sxy[d].sum(0)
        else:
            zd = z[d]
            sx = np.concatenate([self.sx[d].sum(0), zd @ self.sx[d]])
            sy = self.sy[d].sum()
            A = self.sxx[d].sum(0)
            B = np.tensordot(zd, self.sxx[d], 1)
            C = np.tensordot(zd * zd, self.sxx[d], 1)
            sxx = np.block([[A, B], [B, C]])
            sxy = np.concatenate([self.sxy[d].sum(0), zd @ self.sxy[d]])
        xm, ym = sx / sw, sy / sw
        G = sxx / sw - np.outer(xm, xm)
        c = sxy / sw - xm * ym
        return xm, ym, G, c


class _Base:
    coef_: np.ndarray
    intercept_: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_

    def fit(self, X, y, w=None):
        return self.fit_gram(*moments(X, y, w))

    @staticmethod
    def _std(xm, ym, G, c):
        sd = np.sqrt(np.clip(np.diag(G), 1e-12, None))
        return sd, G / np.outer(sd, sd), c / sd

    def _finish(self, b_std, xm, ym, sd):
        self.coef_ = b_std / sd
        self.intercept_ = float(ym - xm @ self.coef_)
        return self


class Ridge(_Base):
    """min 1/n Σ w(y - Xb)^2 + lam*|b|^2  (X standardisé). Solution fermée."""

    def __init__(self, lam: float = 1.0):
        self.lam = lam

    def fit_gram(self, xm, ym, G, c):
        sd, G, c = self._std(xm, ym, G, c)
        b = np.linalg.solve(G + self.lam * np.eye(len(c)), c)
        return self._finish(b, xm, ym, sd)

    @staticmethod
    def path(X, y, lams, w=None) -> np.ndarray:
        """Coefs pour toute une grille de lambdas avec UNE seule décomposition propre."""
        xm, ym, G, c = moments(X, y, w)
        sd = np.sqrt(np.clip(np.diag(G), 1e-12, None))
        d, V = np.linalg.eigh(G / np.outer(sd, sd))
        Vc = V.T @ (c / sd)
        return np.array([(V @ (Vc / (d + lam))) / sd for lam in lams])


class ElasticNet(_Base):
    """1/(2n)||y-Xb||^2 + lam*(alpha|b|_1 + (1-alpha)/2|b|^2). alpha=1 -> Lasso.

    positive=True : contraint les coefs >= 0 (utile quand on a orienté les
    signaux à l'avance — régularisation très efficace contre le sur-ajustement).
    """

    def __init__(self, lam: float = 0.01, alpha: float = 0.5, positive: bool = False,
                 max_iter: int = 2000, tol: float = 1e-7):
        self.lam, self.alpha, self.positive = lam, alpha, positive
        self.max_iter, self.tol = max_iter, tol
        self._warm: np.ndarray | None = None

    def fit_gram(self, xm, ym, G, c):
        sd, G, c = self._std(xm, ym, G, c)
        b0 = self._warm if self._warm is not None and len(self._warm) == len(c) else np.zeros(len(c))
        b = enet_cd_gram(G, c, self.lam, self.alpha, b0, self.positive, self.max_iter, self.tol)
        self._warm = b
        return self._finish(b, xm, ym, sd)

    @staticmethod
    def lam_max(X, y, alpha=1.0, w=None) -> float:
        """Plus petit lambda qui annule tous les coefs (pour construire la grille)."""
        xm, ym, G, c = moments(X, y, w)
        sd = np.sqrt(np.clip(np.diag(G), 1e-12, None))
        return float(np.max(np.abs(c / sd)) / max(alpha, 1e-3))


class Lasso(ElasticNet):
    def __init__(self, lam: float = 0.01, positive: bool = False, **kw):
        super().__init__(lam=lam, alpha=1.0, positive=positive, **kw)


class HuberRidge(_Base):
    """Ridge à perte de Huber par IRLS : les observations extrêmes (news, gaps)
    sont sous-pondérées au lieu de tirer les coefs. delta en unités de MAD."""

    def __init__(self, lam: float = 1.0, delta: float = 1.345, n_iter: int = 4):
        self.lam, self.delta, self.n_iter = lam, delta, n_iter

    def fit(self, X, y, w=None):
        base = np.ones(len(y)) if w is None else w
        ww = base.copy()
        m = Ridge(self.lam)
        for _ in range(self.n_iter):
            m.fit(X, y, ww)
            r = y - m.predict(X)
            s = 1.4826 * np.median(np.abs(r[:: max(1, len(r) // 50000)])) + 1e-12  # MAD (sous-échant.)
            a = np.abs(r) / (self.delta * s)
            ww = base * np.where(a <= 1.0, 1.0, 1.0 / np.maximum(a, 1e-12))
        self.coef_, self.intercept_ = m.coef_, m.intercept_
        self.weights_ = ww
        return self
