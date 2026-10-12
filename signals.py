"""Bibliothèque de signaux (expressions Polars, calculées par symbole).

Entrée : panel long trié (symbol, ts) avec open, high, low, close, volume,
et la table marché jointe (mkt_ret). Aucun signal n'utilise d'info > t.

Le SIGNE des signaux n'a pas d'importance : la régression l'apprend.
On ajoute donc des signaux "bruts" et on laisse ridge/lasso trier.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import polars as pl

from .kernels import offsets_from, rolling_ols_resid

S = "symbol"
R = pl.col("ret")
LP = pl.col("logp")


def _o(e: pl.Expr) -> pl.Expr:
    return e.over(S)


# ---------- briques de base -------------------------------------------------
def add_base_columns(df: pl.DataFrame, beta_window: int = 60) -> pl.DataFrame:
    """ret (log), logp, dollar volume, surprise de volume, beta et résidu vs marché."""
    df = df.sort(S, "ts").with_columns(
        logp=pl.col("close").log(),
        ret=_o(pl.col("close").log().diff()),
        dvol=pl.col("close") * pl.col("volume"),
        lvol=(pl.col("volume") + 1.0).log(),
    ).with_columns(
        vsurp=pl.col("lvol") - _o(pl.col("lvol").rolling_mean(20).shift(1)),
    )
    if "mkt_ret" in df.columns:
        off = offsets_from(df, S)
        y = df["ret"].fill_null(np.nan).to_numpy().astype(np.float64)
        x = df["mkt_ret"].fill_null(np.nan).to_numpy().astype(np.float64)
        beta, resid = rolling_ols_resid(y, x, off, beta_window, beta_window // 2)
        df = df.with_columns(beta=pl.Series(beta).fill_nan(None),
                             resid=pl.Series(resid).fill_nan(None))
    return df


# ---------- signaux ---------------------------------------------------------
def momentum(lb: int, skip: int = 1) -> pl.Expr:
    """Rendement de t-lb à t-skip, normalisé par la vol."""
    return _o((LP.shift(skip) - LP.shift(lb)) / (R.rolling_std(lb) * np.sqrt(lb - skip)))


def reversal(k: int) -> pl.Expr:
    return _o(-R.rolling_sum(k) / (R.rolling_std(60) * np.sqrt(k)))


def resid_reversal(k: int) -> pl.Expr:
    """Retournement du résidu idiosyncratique (vs marché) : le classique stat-arb."""
    r = pl.col("resid")
    return _o(-r.rolling_sum(k) / (r.rolling_std(60) * np.sqrt(k)))


def ma_gap(w: int) -> pl.Expr:
    """Écart prix / moyenne mobile, en unités de vol."""
    return _o((LP - LP.rolling_mean(w)) / (R.rolling_std(w) * np.sqrt(w)))


def ma_cross(fast: int, slow: int) -> pl.Expr:
    return _o((LP.rolling_mean(fast) - LP.rolling_mean(slow)) / (R.rolling_std(slow) * np.sqrt(slow)))


def ewma_cross(fast: int, slow: int) -> pl.Expr:
    return _o((LP.ewm_mean(span=fast) - LP.ewm_mean(span=slow)) / (R.rolling_std(slow) * np.sqrt(slow)))


def vol_ratio(short: int, long: int) -> pl.Expr:
    return _o((R.rolling_std(short) / R.rolling_std(long)).log())


def volume_surprise(w: int) -> pl.Expr:
    return _o(pl.col("vsurp").rolling_mean(w))


def signed_volume(w: int) -> pl.Expr:
    """Rendement signé pondéré par la surprise de volume (pression acheteuse/vendeuse)."""
    return _o((R.sign() * pl.col("vsurp").clip(0, None)).rolling_mean(w))


def range_position(w: int) -> pl.Expr:
    lo = pl.col("low").rolling_min(w)
    hi = pl.col("high").rolling_max(w)
    return _o((pl.col("close") - lo) / (hi - lo) - 0.5)


def amihud(w: int) -> pl.Expr:
    return _o((R.abs() / pl.col("dvol")).rolling_mean(w).log())


def parkinson_vol(w: int) -> pl.Expr:
    hl = (pl.col("high") / pl.col("low")).log() ** 2
    return _o((hl.rolling_mean(w) / (4 * np.log(2))).sqrt().log())


def beta_signal() -> pl.Expr:
    return pl.col("beta")


DEFAULT_SIGNALS: dict[str, Callable[[], pl.Expr]] = {
    "mom_60_5": lambda: momentum(60, 5),
    "mom_20_1": lambda: momentum(20, 1),
    "rev_1": lambda: reversal(1),
    "rev_5": lambda: reversal(5),
    "rrev_1": lambda: resid_reversal(1),
    "rrev_5": lambda: resid_reversal(5),
    "ma_gap_10": lambda: ma_gap(10),
    "ma_cross_5_20": lambda: ma_cross(5, 20),
    "ewma_cross_10_50": lambda: ewma_cross(10, 50),
    "vol_ratio_5_60": lambda: vol_ratio(5, 60),
    "vsurp_5": lambda: volume_surprise(5),
    "svol_5": lambda: signed_volume(5),
    "range_pos_20": lambda: range_position(20),
    "amihud_20": lambda: amihud(20),
    "pk_vol_20": lambda: parkinson_vol(20),
    "beta": beta_signal,
}


def compute_signals(df: pl.DataFrame, signals: dict[str, Callable[[], pl.Expr]] | None = None,
                    beta_window: int = 60) -> tuple[pl.DataFrame, list[str]]:
    signals = signals or DEFAULT_SIGNALS
    df = add_base_columns(df, beta_window)
    df = df.with_columns([f().alias(name) for name, f in signals.items()])
    # inf -> null (divisions par vol nulle, etc.)
    df = df.with_columns([
        pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c) for c in signals
    ])
    return df, list(signals)
