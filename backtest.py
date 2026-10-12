"""Backtest long/short dollar-neutre vectorisé, avec coûts et lissage du turnover.

Poids en t (connus à la clôture t), appliqués au rendement ret1_fwd
(t+lag -> t+lag+1). Coûts = cost_bps * Σ|Δw|.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .validation.metrics import sharpe_stats


def prediction_weights(preds: pl.DataFrame, pred: str = "pred", halflife: float | None = None,
                       gross: float = 1.0) -> pl.DataFrame:
    """w = (pred - moyenne)/Σ|.| par date ; option lissage EWMA par titre (réduit le turnover)."""
    df = preds.with_columns(_c=pl.col(pred) - pl.col(pred).mean().over("ts"))
    df = df.with_columns(w=gross * pl.col("_c") / pl.col("_c").abs().sum().over("ts"))
    if halflife:
        df = df.sort("symbol", "ts").with_columns(w=pl.col("w").ewm_mean(half_life=halflife).over("symbol"))
        df = df.with_columns(_c=pl.col("w") - pl.col("w").mean().over("ts"))
        df = df.with_columns(w=gross * pl.col("_c") / pl.col("_c").abs().sum().over("ts"))
    return df.drop("_c")


def run_backtest(preds: pl.DataFrame, pred: str = "pred", cost_bps: float = 2.0,
                 halflife: float | None = None, bars_per_year: int = 252) -> tuple[pl.DataFrame, dict]:
    """preds doit contenir ts, symbol, pred, ret1_fwd (log-rendement de la barre suivante)."""
    w = prediction_weights(preds, pred, halflife).sort("symbol", "ts")
    w = w.with_columns(dw=(pl.col("w") - pl.col("w").shift(1).over("symbol").fill_null(0.0)).abs())
    pnl = (w.group_by("ts")
           .agg(gross=(pl.col("w") * (pl.col("ret1_fwd").exp() - 1)).sum(),
                turnover=pl.col("dw").sum())
           .sort("ts")
           .with_columns(net=pl.col("gross") - cost_bps * 1e-4 * pl.col("turnover")))
    stats = sharpe_stats(pnl["net"].to_numpy(), bars_per_year)
    stats["sharpe_gross"] = sharpe_stats(pnl["gross"].to_numpy(), bars_per_year)["sharpe_ann"]
    stats["turnover"] = float(pnl["turnover"].mean())
    return pnl, stats
