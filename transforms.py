"""Transformations en coupe (par date), cibles, régimes de marché, univers point-in-time."""
from __future__ import annotations

import numpy as np
import polars as pl

T, S = "ts", "symbol"


def cs_robust_zscore(df: pl.DataFrame, cols: list[str], by: str | list[str] = T,
                     clip: float = 3.0, fill: float | None = 0.0) -> pl.DataFrame:
    """Z-score robuste en coupe : (x - médiane)/(1.4826*MAD), écrêté, puis re-standardisé.

    Résiste aux outliers (splits ratés, prints aberrants) bien mieux qu'un z-score classique.
    `fill=0` : un signal manquant = neutre (pas de pari).
    """
    med = {c: f"__med_{c}" for c in cols}
    df = df.with_columns([pl.col(c).median().over(by).alias(med[c]) for c in cols])
    df = df.with_columns([
        ((pl.col(c) - pl.col(med[c])) /
         (1.4826 * (pl.col(c) - pl.col(med[c])).abs().median().over(by) + 1e-12)).clip(-clip, clip).alias(c)
        for c in cols
    ]).drop(list(med.values()))
    df = df.with_columns([
        ((pl.col(c) - pl.col(c).mean().over(by)) / (pl.col(c).std().over(by) + 1e-12)).alias(c)
        for c in cols
    ])
    if fill is not None:
        df = df.with_columns([pl.col(c).fill_null(fill).fill_nan(fill) for c in cols])
    return df


def cs_rank(df: pl.DataFrame, cols: list[str], by: str | list[str] = T) -> pl.DataFrame:
    """Rang uniforme centré dans [-0.5, 0.5] — la transformation la plus robuste."""
    return df.with_columns([
        ((pl.col(c).rank("average").over(by) - 0.5) / pl.col(c).count().over(by) - 0.5)
        .fill_null(0.0).alias(c) for c in cols
    ])


def neutralize(df: pl.DataFrame, cols: list[str], group: str) -> pl.DataFrame:
    """Neutralisation par groupe (secteur, industrie...) à chaque date."""
    return df.with_columns([(pl.col(c) - pl.col(c).mean().over([T, group])).alias(c) for c in cols])


def add_targets(df: pl.DataFrame, horizons=(1, 5), lag: int = 1, vol_window: int = 20,
                clip: float = 5.0) -> pl.DataFrame:
    """Cibles forward, SANS fuite :

    - le signal est connu en t (clôture), on exécute en t+lag ;
    - fwd_h  = log(P[t+lag+h] / P[t+lag])           (rendement brut, pour l'évaluation)
    - y_h    = fwd_h démeané en coupe / (vol_t * sqrt(h)), écrêté   (cible d'entraînement)
    - ret1_fwd = rendement de la barre t+lag -> t+lag+1 (pour le backtest)

    Vol-scaler la cible = moindres carrés pondérés implicites : les titres très
    volatils ne dominent plus la régression (robustesse au bruit).
    """
    lp = pl.col("logp")
    df = df.sort(S, T).with_columns(
        __vol=pl.col("ret").rolling_std(vol_window).over(S),
        ret1_fwd=(lp.shift(-lag - 1) - lp.shift(-lag)).over(S),
    )
    df = df.with_columns([(lp.shift(-lag - h) - lp.shift(-lag)).over(S).alias(f"fwd_{h}") for h in horizons])
    df = df.with_columns([
        ((pl.col(f"fwd_{h}") - pl.col(f"fwd_{h}").mean().over(T)) / (pl.col("__vol") * np.sqrt(h)))
        .clip(-clip, clip).alias(f"y_{h}") for h in horizons
    ]).drop("__vol")
    return df.with_columns([
        pl.when(pl.col(f"y_{h}").is_finite()).then(pl.col(f"y_{h}")).otherwise(None).alias(f"y_{h}")
        for h in horizons
    ])


def add_market_regime(market: pl.DataFrame, vol_window: int = 20, norm_window: int = 250,
                      thresholds=(-0.5, 0.5)) -> pl.DataFrame:
    """Variables de conditionnement à partir de la série marché (ts, mkt_ret[, vix]).

    vol_z  : vol réalisée du marché, standardisée sur une fenêtre PASSÉE glissante
    regime : 0 = calme, 1 = normal, 2 = stress   (seuils sur vol_z)
    Si `vix` est présent, on ajoute vix_z de la même façon.
    """
    m = market.sort(T).with_columns(mkt_vol=pl.col("mkt_ret").rolling_std(vol_window))
    m = m.with_columns(
        vol_z=((pl.col("mkt_vol") - pl.col("mkt_vol").rolling_mean(norm_window, min_samples=60))
               / pl.col("mkt_vol").rolling_std(norm_window, min_samples=60)).clip(-4, 4),
        mkt_trend=pl.col("mkt_ret").rolling_sum(60) / (pl.col("mkt_vol") * np.sqrt(60)),
    )
    if "vix" in m.columns:
        lv = pl.col("vix").log()
        m = m.with_columns(vix_z=((lv - lv.rolling_mean(norm_window, min_samples=60))
                                  / lv.rolling_std(norm_window, min_samples=60)).clip(-4, 4))
    lo, hi = thresholds
    return m.with_columns(
        regime=pl.when(pl.col("vol_z") < lo).then(0).when(pl.col("vol_z") > hi).then(2)
        .when(pl.col("vol_z").is_not_null()).then(1).otherwise(None).cast(pl.Int32)
    )


def market_from_universe(bars: pl.DataFrame) -> pl.DataFrame:
    """Proxy marché équipondéré si on n'a pas d'indice."""
    return (bars.sort(S, T)
            .with_columns(r=pl.col("close").log().diff().over(S))
            .group_by(T).agg(mkt_ret=pl.col("r").mean()).sort(T))


def liquidity_universe(df: pl.DataFrame, top_n: int = 500, adv_window: int = 20,
                       min_price: float = 5.0) -> pl.DataFrame:
    """Univers point-in-time : top N par dollar-volume moyen, calculé avec des données <= t-1.

    Évite le biais du survivant d'un univers "constituants actuels".
    """
    df = df.sort(S, T).with_columns(
        __adv=(pl.col("close") * pl.col("volume")).rolling_mean(adv_window).shift(1).over(S),
        __px=pl.col("close").shift(1).over(S),
    )
    return df.with_columns(
        in_univ=(pl.col("__adv").rank("ordinal", descending=True).over(T) <= top_n)
        & (pl.col("__px") >= min_price) & pl.col("__adv").is_not_null()
    ).with_columns(pl.col("in_univ").fill_null(False)).drop("__adv", "__px")
