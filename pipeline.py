"""Walk-forward : sélection des hyperparamètres en interne (purgée), ré-entraînement
à chaque pli, prédictions strictement hors-échantillon.

Chemin rapide : si le modèle expose `fit_gram` (Ridge, Lasso, ElasticNet,
RegimeRidge, InteractionModel) et que la variable de conditionnement est
définie au niveau de la date (régime/vol marché), on n'utilise que les
moments par date -> chaque fit coûte O(D·p²), indépendamment du nombre d'actions.
Sinon (HuberRidge, conditionnement par titre) : chemin ligne à ligne.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import polars as pl

from .features.kernels import grouped_ic, offsets_from
from .models.linear import MomentCache
from .validation.cv import PurgedWalkForward


@dataclass
class ModelSpec:
    name: str
    make: Callable[[float], Any]           # lam -> modèle
    grid: Sequence[float]
    kind: str = "plain"                    # "plain" | "regime" | "interaction"
    cond_col: str | None = None            # colonne régime (int) ou z (float)
    history: list = field(default_factory=list)


class _Ctx:
    """Données alignées (triées ts, symbol) + caches."""

    def __init__(self, ds, features, target, weight_col):
        self.off = offsets_from(ds, "ts")
        self.X = ds.select(features).to_numpy().astype(np.float64)
        self.y = ds[target].to_numpy().astype(np.float64)
        self.w = ds[weight_col].to_numpy().astype(np.float64) if weight_col else None
        self.ds = ds
        self._cache: MomentCache | None = None
        self._date_level: dict[str, np.ndarray | None] = {}

    @property
    def cache(self) -> MomentCache:
        if self._cache is None:
            self._cache = MomentCache(self.X, self.y, self.w, self.off)
        return self._cache

    def rows(self, d):
        return slice(self.off[d[0]], self.off[d[-1] + 1])

    def sub_off(self, d):
        return self.off[d[0]: d[-1] + 2] - self.off[d[0]]

    def cond_rows(self, spec):
        if spec.kind == "regime":
            return self.ds[spec.cond_col].fill_null(-1).to_numpy().astype(np.int64)
        if spec.kind == "interaction":
            return self.ds[spec.cond_col].fill_null(0.0).to_numpy().astype(np.float64)
        return None

    def cond_dates(self, spec, c_rows):
        """Valeur par date si la variable est constante dans chaque date, sinon None."""
        if spec.cond_col not in self._date_level:
            g = (self.ds.select("ts", spec.cond_col).group_by("ts", maintain_order=True)
                 .agg(pl.col(spec.cond_col).n_unique().alias("u")))
            const = bool((g["u"] <= 1).all())
            self._date_level[spec.cond_col] = c_rows[self.off[:-1]] if const else None
        return self._date_level[spec.cond_col]


def _fit(spec, m, ctx: _Ctx, d, c_rows, c_dates):
    fast = hasattr(m, "fit_gram") or hasattr(m, "fit_moments")
    if fast and spec.kind == "plain":
        return m.fit_gram(*ctx.cache.moments(d))
    if fast and spec.kind == "interaction" and c_dates is not None:
        return m.fit_gram(*ctx.cache.moments(d, z=c_dates))
    if fast and spec.kind == "regime" and c_dates is not None:
        rd = c_dates[d]
        by = {r: (ctx.cache.moments(d[rd == r]), ctx.cache.n_rows(d[rd == r]))
              for r in np.unique(rd) if r >= 0}
        return m.fit_moments(ctx.cache.moments(d), by)
    r = ctx.rows(d)
    w = None if ctx.w is None else ctx.w[r]
    if spec.kind == "plain":
        return m.fit(ctx.X[r], ctx.y[r], w)
    return m.fit(ctx.X[r], ctx.y[r], c_rows[r], w)


def _predict(spec, m, ctx: _Ctx, d, c_rows):
    r = ctx.rows(d)
    return m.predict(ctx.X[r]) if spec.kind == "plain" else m.predict(ctx.X[r], c_rows[r])


def walk_forward(ds: pl.DataFrame, features: list[str], target: str, specs: list[ModelSpec],
                 cv: PurgedWalkForward, weight_col: str | None = None,
                 keep_cols: Sequence[str] = ("ret1_fwd",), verbose: bool = True) -> pl.DataFrame:
    ds = ds.filter(pl.col(target).is_not_null()).sort("ts", "symbol")
    ctx = _Ctx(ds, features, target, weight_col)
    n_dates = len(ctx.off) - 1
    preds = {s.name: np.full(len(ctx.y), np.nan) for s in specs}
    conds = {}
    for s in specs:
        cr = ctx.cond_rows(s)
        conds[s.name] = (cr, ctx.cond_dates(s, cr) if cr is not None else None)

    t0 = time.perf_counter()
    for k, (tr, te) in enumerate(cv.split(n_dates)):
        itr, iva = cv.inner_split(tr)
        o_iva = ctx.sub_off(iva)
        y_iva = ctx.y[ctx.rows(iva)]
        for s in specs:
            c_rows, c_dates = conds[s.name]
            best, best_ic = s.grid[0], np.nan
            if len(s.grid) > 1:                     # 1) sélection de lambda (IC de validation)
                best_ic = -np.inf
                for lam in s.grid:
                    m = _fit(s, s.make(lam), ctx, itr, c_rows, c_dates)
                    ic = np.nanmean(grouped_ic(_predict(s, m, ctx, iva, c_rows), y_iva, o_iva, True, 20))
                    if ic > best_ic:
                        best, best_ic = lam, ic
            m = _fit(s, s.make(best), ctx, tr, c_rows, c_dates)   # 2) refit sur tout le train
            preds[s.name][ctx.rows(te)] = _predict(s, m, ctx, te, c_rows)
            s.history.append({"fold": k, "test_start": ds["ts"][int(ctx.off[te[0]])], "lam": best,
                              "val_ic": best_ic, "model": m})
        if verbose:
            print(f"pli {k:3d}  train={len(tr):5d} dates  test={len(te):3d}  "
                  f"{time.perf_counter() - t0:6.1f}s", end="\r")
    if verbose:
        print()
    out = ds.select(["ts", "symbol", target, *[c for c in keep_cols if c in ds.columns]])
    out = out.with_columns([pl.Series(f"pred_{n}", p) for n, p in preds.items()])
    return out.filter(pl.any_horizontal([pl.col(f"pred_{n}").is_not_nan() for n in preds]))


def coef_table(spec: ModelSpec, names: list[str], last_n: int = 10) -> pl.DataFrame:
    """Coefs moyens et stabilité (fraction de plis où le signe est celui de la moyenne)."""
    hist = spec.history[-last_n:]
    if spec.kind == "regime":
        B = np.array([h["model"].pool_.coef_ for h in hist])
    elif spec.kind == "interaction":
        B = np.array([h["model"].base.coef_ for h in hist])
        names = names + [f"{n}*z" for n in names]
    else:
        B = np.array([h["model"].coef_ for h in hist])
    mean = B.mean(0)
    stab = (np.sign(B) == np.sign(mean)).mean(0)
    return (pl.DataFrame({"feature": names[: B.shape[1]], "coef": mean, "sign_stability": stab})
            .sort(pl.col("coef").abs(), descending=True))
