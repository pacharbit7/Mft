"""Pipeline complet sur données synthétiques (aucune connexion IBKR requise).

    python examples/demo_synthetic.py            # alphas plantés : doit trouver
    python examples/demo_synthetic.py --null     # aucun alpha, 8 graines : Sharpe ~ 0 attendu
"""
from __future__ import annotations

import sys
import time

import numpy as np
import polars as pl

from mftlab.backtest import run_backtest
from mftlab.data.synthetic import make_synthetic
from mftlab.features.signals import compute_signals
from mftlab.features.transforms import (add_market_regime, add_targets, cs_robust_zscore,
                                        liquidity_universe)
from mftlab.models.conditional import InteractionModel, RegimeRidge
from mftlab.models.linear import ElasticNet, HuberRidge, Lasso, Ridge
from mftlab.pipeline import ModelSpec, coef_table, walk_forward
from mftlab.validation.cv import PurgedWalkForward
from mftlab.validation.metrics import (deflated_sharpe, ic_by_regime, ic_by_signal, ic_series,
                                      ic_summary)

pl.Config.set_tbl_rows(30)
pl.Config.set_float_precision(4)

H, LAG = 1, 1
TARGET = f"y_{H}"
null = "--null" in sys.argv



def build_panel(seed: int = 0, alpha_scale: float = 1.0):
    bars, market = make_synthetic(n_symbols=300, n_bars=2500, seed=seed, alpha_scale=alpha_scale)
    market = add_market_regime(market)
    panel = bars.join(market.select("ts", "mkt_ret", "vol_z", "vix_z", "regime"), on="ts", how="left")
    panel, feats = compute_signals(panel)
    panel = add_targets(panel, horizons=(1, 5), lag=LAG)          # cibles AVANT filtrage univers
    panel = liquidity_universe(panel, top_n=250).filter(pl.col("in_univ"))
    panel = cs_robust_zscore(panel, feats)                        # z-score robuste en coupe
    return panel, feats


if null:
    # Distribution sous H0 : sans alpha, le Sharpe OOS doit être centré sur 0.
    cv = PurgedWalkForward(min_train=750, test_size=63, horizon=H, lag=LAG, embargo=5)
    out = []
    for seed in range(8):
        panel, feats = build_panel(seed, alpha_scale=0.0)
        p = walk_forward(panel, feats, TARGET, [ModelSpec("ridge", lambda l: Ridge(l), [1, 10, 100])],
                         cv, verbose=False)
        st = run_backtest(p, "pred_ridge", cost_bps=0)[1]
        ic = ic_summary(ic_series(p, "pred_ridge", TARGET)["ic"].to_numpy(), H)
        out.append({"seed": seed, "sharpe_gross": st["sharpe_ann"], "ic_t_nw": ic["t_nw"]})
        print(out[-1])
    df = pl.DataFrame(out)
    print(df.select(pl.col("sharpe_gross").mean().alias("sharpe_moyen"),
                    pl.col("sharpe_gross").std().alias("sharpe_std")))
    sys.exit(0)

t0 = time.perf_counter()
panel, feats = build_panel(seed=0)
print(f"features : {panel.shape}  {time.perf_counter() - t0:.1f}s")

# 2) diagnostic univarié (IC de chaque signal, t-stat Newey-West) — sur la 1re moitié seulement
cut = panel["ts"].unique().sort()[panel["ts"].n_unique() // 2]
print("\nIC univariés (1re moitié) :")
print(ic_by_signal(panel.filter(pl.col("ts") < cut), feats, TARGET, H)
      .select("signal", "ic_mean", "t_nw", "hit"))

print("\nIC conditionnel au régime de vol (0 calme / 1 normal / 2 stress), 1re moitié :")
print(ic_by_regime(panel.filter(pl.col("ts") < cut), ["ma_gap_10", "rev_5", "rrev_5", "ewma_cross_10_50"], TARGET))

# 3) modèles
specs = [
    ModelSpec("ridge", lambda l: Ridge(l), [0.1, 1, 10, 100, 1000]),
    ModelSpec("lasso", lambda l: Lasso(l), [1e-4, 3e-4, 1e-3, 3e-3]),
    ModelSpec("enet", lambda l: ElasticNet(l, alpha=0.5), [1e-4, 1e-3, 3e-3]),
    ModelSpec("huber", lambda l: HuberRidge(l), [1, 10, 100]),
    ModelSpec("regime", lambda l: RegimeRidge(lam_pool=10, lam_regime=l), [0.3, 3, 30], "regime", "regime"),
    ModelSpec("inter_vol", lambda l: InteractionModel(Ridge(l)), [1, 10, 100], "interaction", "vol_z"),
]
cv = PurgedWalkForward(min_train=750, test_size=63, horizon=H, lag=LAG, embargo=5)

t1 = time.perf_counter()
preds = walk_forward(panel, feats, TARGET, specs, cv)
print(f"walk-forward : {time.perf_counter() - t1:.1f}s, {preds.height:,} prédictions OOS")

# 4) évaluation hors-échantillon
rows, srs = [], []
for s in specs:
    ic = ic_series(preds, f"pred_{s.name}", TARGET)["ic"].to_numpy()
    _, st = run_backtest(preds, f"pred_{s.name}", cost_bps=2.0, halflife=None)
    srs.append(st["sr_bar"])
    rows.append({"model": s.name, **{k: v for k, v in ic_summary(ic, H).items() if k != "n_dates"},
                 "sharpe_net": st["sharpe_ann"], "sharpe_gross": st["sharpe_gross"],
                 "turnover": st["turnover"], "max_dd": st["max_dd"], "_st": st})
var_sr = float(np.var(srs, ddof=1))
for r in rows:
    st = r.pop("_st")
    r["DSR"] = deflated_sharpe(st["sr_bar"], st["n"], st["skew"], st["kurt"], len(specs), var_sr)
print("\nPerformance hors-échantillon (coûts 2 bps) :")
print(pl.DataFrame(rows))

# 5) interprétabilité
print("\nRidge — coefs moyens des 10 derniers plis :")
print(coef_table(specs[0], feats).head(8))
print("\nLasso — sélection :")
print(coef_table(specs[1], feats).filter(pl.col("coef") != 0))
for name in ("regime", "inter_vol"):
    sp = next(s for s in specs if s.name == name)
    print(f"{name:10s} lambdas choisis (5 derniers plis) :", [h["lam"] for h in sp.history[-5:]])
print(f"\ntotal {time.perf_counter() - t0:.1f}s")
