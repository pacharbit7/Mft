"""Stockage Parquet : un fichier par (type, taille de barre, symbole). Lecture lazy."""
from __future__ import annotations

from pathlib import Path

import polars as pl


class ParquetStore:
    def __init__(self, root: str | Path = "data"):
        self.root = Path(root)

    def _dir(self, kind: str, bar_size: str) -> Path:
        d = self.root / kind / bar_size.replace(" ", "")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write(self, df: pl.DataFrame, kind: str, bar_size: str, symbol: str) -> Path:
        """Fusionne avec l'existant (dédoublonnage sur ts, la dernière version gagne)."""
        p = self._dir(kind, bar_size) / f"{symbol.replace(' ', '_')}.parquet"
        df = df.with_columns(symbol=pl.lit(symbol))
        if p.exists():
            df = pl.concat([pl.read_parquet(p), df], how="diagonal_relaxed")
        df.unique("ts", keep="last").sort("ts").write_parquet(p)
        return p

    def last_ts(self, kind: str, bar_size: str, symbol: str):
        p = self._dir(kind, bar_size) / f"{symbol.replace(' ', '_')}.parquet"
        return pl.read_parquet(p, columns=["ts"])["ts"].max() if p.exists() else None

    def scan(self, kind: str, bar_size: str, symbols: list[str] | None = None) -> pl.LazyFrame:
        lf = pl.scan_parquet(self._dir(kind, bar_size) / "*.parquet")
        return lf.filter(pl.col("symbol").is_in(symbols)) if symbols else lf
