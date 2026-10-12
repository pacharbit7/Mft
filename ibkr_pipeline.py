"""Pipeline réel : IBKR -> Parquet -> features -> walk-forward.

Prérequis : TWS ou IB Gateway lancé, API activée, abonnements données US.
    python examples/ibkr_pipeline.py download    # 1re fois (long : pacing IBKR)
    python examples/ibkr_pipeline.py research
"""
from __future__ import annotations

import datetime as dt
import sys

import polars as pl

from mftlab.backtest import run_backtest
from mftlab.data.ibkr import IBKRData, load_market_table
from mftlab.data.store import ParquetStore
from mftlab.data.universe import sp500_current
from mftlab.features.signals import compute_signals
from mftlab.features.transforms import (add_market_regime, add_targets, cs_robust_zscore,
                                        liquidity_universe, neutralize)
from mftlab.models.conditional import RegimeRidge
from mftlab.models.linear import Lasso, Ridge
from mftlab.pipeline import ModelSpec, coef_table, walk_forward
from mftlab.validation.cv import PurgedWalkForward
from mftlab.validation.metrics import ic_by_signal, ic_series, ic_summary

BAR = "1 hour"            # MFT : "1 hour", "30 mins", "15 mins"...
BARS_PER_YEAR = 252 * 7   # ~7 barres horaires RTH par séance
START = dt.datetime(2022, 1, 1)
store = ParquetStore("data")

if sys.argv[1:] == ["download"]:
    univ = sp500_current()                         # ATTENTION biais du survivant (cf. universe.py)
    univ.write_parquet("data/universe.parquet")
    with IBKRData(port=7497) as ib:                # 4002 si IB Gateway paper
        ib.download_market_context(START, BAR, store)
        failed = ib.download(univ["ib_symbol"].to_list(), START, BAR, store)
    print("échecs :", failed)
    sys.exit(0)

# ---------------- recherche ----------------
H, LAG = 7, 1                                       # horizon 1 séance, exécution barre suivante
univ = pl.read_parquet("data/universe.parquet").select(symbol="ib_symbol", sector="sector")
bars = store.scan("stk", BAR).collect().join(univ, on="symbol", how="left")
market = add_market_regime(load_market_table(store, BAR, proxy="SPY", vix="VIX"),
                           vol_window=35, norm_window=1750)

panel = bars.join(market.select("ts", "mkt_ret", "vol_z", "vix_z", "regime"), on="ts", how="left")
panel, feats = compute_signals(panel, beta_window=350)
panel = add_targets(panel, horizons=(1, H), lag=LAG)
panel = liquidity_universe(panel, top_n=400, adv_window=140).filter(pl.col("in_univ"))
panel = cs_robust_zscore(panel, feats)
panel = neutralize(panel, feats, "sector")          # signaux neutres secteur
target = f"y_{H}"

print(ic_by_signal(panel, feats, target, H))
specs = [
    ModelSpec("ridge", lambda l: Ridge(l), [0.1, 1, 10, 100, 1000]),
    ModelSpec("lasso", lambda l: Lasso(l), [1e-4, 3e-4, 1e-3, 3e-3]),
    ModelSpec("regime", lambda l: RegimeRidge(10, l), [0.3, 3, 30], "regime", "regime"),
]
cv = PurgedWalkForward(min_train=7 * 250, test_size=7 * 20, horizon=H, lag=LAG, embargo=7)
preds = walk_forward(panel, feats, target, specs, cv)
for s in specs:
    ic = ic_summary(ic_series(preds, f"pred_{s.name}", target)["ic"].to_numpy(), H, BARS_PER_YEAR)
    _, st = run_backtest(preds, f"pred_{s.name}", cost_bps=3, halflife=H, bars_per_year=BARS_PER_YEAR)
    print(s.name, ic, {k: round(v, 3) for k, v in st.items() if isinstance(v, float)})
print(coef_table(specs[0], feats))
