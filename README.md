# mftlab — lab d'arbitrage statistique moyenne fréquence

Polars pour les données et features, Numba pour les noyaux chauds, IBKR (`ib_async`) pour les données.

```
pip install -r requirements.txt
PYTHONPATH=. python examples/demo_synthetic.py          # pipeline complet, alphas plantés
PYTHONPATH=. python examples/demo_synthetic.py --null   # contrôle : 8 marchés sans alpha
PYTHONPATH=. python examples/ibkr_pipeline.py download  # données réelles
PYTHONPATH=. python examples/ibkr_pipeline.py research
```

## Architecture

| Module | Rôle |
|---|---|
| `data/ibkr.py` | historique par blocs avec pacing, indices (SPX, VIX, NDX, RUT), ETF marché/secteurs |
| `data/universe.py` | compo S&P 500 / Nasdaq-100, appartenance historique approx., filtre point-in-time |
| `data/store.py` | Parquet par symbole, écriture incrémentale, lecture lazy |
| `features/signals.py` | 16 signaux (momentum, MA, EWMA, reversal, résidu vs marché, volume, range, Amihud, vol) |
| `features/transforms.py` | z-score robuste en coupe (médiane/MAD), rang, neutralisation secteur, cibles, régimes |
| `features/kernels.py` | Numba : beta/résidus glissants, IC par date, coordinate descent, moments par date |
| `models/linear.py` | Ridge (+ chemin par eigh), Lasso / ElasticNet (option coefs ≥ 0), Huber-Ridge |
| `models/conditional.py` | RegimeRidge (shrinkage vers le pool), modèle à interactions b(z) = b + g·z |
| `validation/cv.py` | walk-forward purgé + embargo |
| `validation/metrics.py` | IC Spearman, t-stat Newey-West, IC par régime, Sharpe, Deflated Sharpe |
| `backtest.py` | L/S dollar-neutre, coûts, lissage EWMA des poids |

## Règles anti-biais intégrées
1. Signal connu à la clôture t, exécution à t+`lag` ; cibles calculées **avant** le filtrage d'univers.
2. Univers par liquidité calculé en t-1 (pas de biais du survivant).
3. Purge `lag + horizon + embargo` entre train et test, y compris pour la validation interne des lambdas.
4. IC jugé par t-stat Newey-West (IC chevauchants = autocorrélés).
5. Deflated Sharpe : passer le **nombre total** de configurations essayées, pas seulement celles affichées.
6. Toujours lancer le contrôle `--null` après avoir modifié un signal ou une cible.
