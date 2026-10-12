"""Composition d'indices / univers.

IBKR ne donne pas les constituants. Options :
 1. `sp500_current()` / `nasdaq100_current()` : scrape Wikipedia. ATTENTION biais du
    survivant si on backteste le passé avec la compo actuelle.
 2. `sp500_membership_history()` : reconstruit une appartenance point-in-time
    approximative à partir du tableau des changements Wikipedia.
 3. Le plus propre : un fichier d'appartenance historique (Norgate, CRSP, Compustat…)
    au format (symbol, start, end) chargé via `apply_membership`.
 4. Alternative sans fournisseur : `liquidity_universe` (features/transforms.py).

Mapping de tickers vers IBKR : "BRK.B" -> "BRK B".
"""
from __future__ import annotations

import io

import polars as pl

_UA = {"User-Agent": "Mozilla/5.0 (mftlab research)"}


def _read_html_tables(url: str):
    import pandas as pd  # dépendances optionnelles : pandas, lxml, requests
    import requests
    html = requests.get(url, headers=_UA, timeout=30).text
    return pd.read_html(io.StringIO(html))


def to_ib_symbol(s: str) -> str:
    return s.replace(".", " ").replace("-", " ")


def sp500_current() -> pl.DataFrame:
    t = _read_html_tables("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    df = pl.from_pandas(t[["Symbol", "GICS Sector", "GICS Sub-Industry"]])
    return df.rename({"Symbol": "symbol", "GICS Sector": "sector", "GICS Sub-Industry": "industry"}) \
             .with_columns(ib_symbol=pl.col("symbol").map_elements(to_ib_symbol, return_dtype=pl.Utf8))


def nasdaq100_current() -> pl.DataFrame:
    for t in _read_html_tables("https://en.wikipedia.org/wiki/Nasdaq-100"):
        cols = {c.lower(): c for c in t.columns.astype(str)}
        if "ticker" in cols:
            df = pl.from_pandas(t.astype(str)).rename({cols["ticker"]: "symbol"})
            return df.with_columns(ib_symbol=pl.col("symbol").map_elements(to_ib_symbol, return_dtype=pl.Utf8))
    raise RuntimeError("table Nasdaq-100 introuvable")


def sp500_membership_history() -> pl.DataFrame:
    """Appartenance approximative (symbol, start, end) en rejouant à rebours les changements.

    Approximation : les changements Wikipedia sont incomplets avant ~2000 et les
    tickers renommés ne sont pas suivis. Suffisant pour réduire fortement le biais
    du survivant, pas pour de la recherche publiable.
    """
    import pandas as pd
    tables = _read_html_tables("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    current = set(tables[0]["Symbol"].astype(str))
    ch = tables[1].copy()
    ch.columns = ["date", "added", "added_name", "removed", "removed_name", "reason"][: len(ch.columns)]
    ch["date"] = pd.to_datetime(ch["date"], errors="coerce")
    ch = ch.dropna(subset=["date"]).sort_values("date", ascending=False)
    members, end_of, rows = set(current), {s: None for s in current}, []
    for _, r in ch.iterrows():
        a, rm = str(r["added"]), str(r["removed"])
        if a and a != "nan" and a in members:          # avant cette date, `a` n'y était pas
            rows.append((a, r["date"], end_of.pop(a, None)))
            members.discard(a)
        if rm and rm != "nan" and rm not in members:   # avant cette date, `rm` y était
            members.add(rm)
            end_of[rm] = r["date"]
    rows += [(s, None, end_of.get(s)) for s in members]
    return pl.DataFrame(rows, schema={"symbol": pl.Utf8, "start": pl.Datetime, "end": pl.Datetime},
                        orient="row")


def apply_membership(panel: pl.DataFrame, membership: pl.DataFrame) -> pl.DataFrame:
    """Ajoute in_index (bool) : ts ∈ [start, end) pour au moins une période du symbole."""
    j = panel.select("ts", "symbol").join(membership, on="symbol", how="inner").filter(
        (pl.col("start").is_null() | (pl.col("ts") >= pl.col("start")))
        & (pl.col("end").is_null() | (pl.col("ts") < pl.col("end")))
    ).select("ts", "symbol").unique().with_columns(in_index=pl.lit(True))
    return panel.join(j, on=["ts", "symbol"], how="left").with_columns(pl.col("in_index").fill_null(False))
