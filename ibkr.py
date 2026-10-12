"""Connecteur Interactive Brokers via `ib_async` (fork maintenu de ib_insync).

    pip install ib_async
    TWS / IB Gateway lancé, API activée. Ports : 7497 TWS paper, 7496 TWS live,
    4002 Gateway paper, 4001 Gateway live.

Points d'attention IBKR :
 - pacing : ~60 requêtes historiques / 10 min pour les petites barres, pas de
   requête identique < 15 s. Le Pacer ci-dessous reste prudent.
 - les indices (SPX, VIX…) n'ont en général PAS de volume : utiliser les ETF
   (SPY, QQQ, IWM, XL*) pour le volume marché/secteur.
 - IBKR ne fournit PAS la composition des indices : voir universe.py.
 - données ajustées des dividendes : whatToShow="ADJUSTED_LAST" (barres >= 1 jour).
"""
from __future__ import annotations

import datetime as dt
import time
from collections import deque

import polars as pl

from .store import ParquetStore

try:
    from ib_async import IB, Contract, Index, Stock
except ImportError:  # pragma: no cover
    IB = None

# durée par requête (prudente) selon la taille de barre
CHUNK = {"1 min": "2 D", "5 mins": "1 W", "15 mins": "2 W", "30 mins": "1 M",
         "1 hour": "1 M", "1 day": "1 Y"}

MARKET_INDEXES = [("SPX", "CBOE"), ("VIX", "CBOE"), ("NDX", "NASDAQ"), ("RUT", "RUSSELL")]
MARKET_ETFS = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU",
               "XLB", "XLRE", "XLC", "TLT", "HYG"]


class Pacer:
    """Limiteur : au plus max_req requêtes par fenêtre `period` s, et min_gap s entre deux."""

    def __init__(self, max_req: int = 55, period: float = 600.0, min_gap: float = 1.0):
        self.max_req, self.period, self.min_gap = max_req, period, min_gap
        self.hist: deque[float] = deque()

    def wait(self):
        now = time.monotonic()
        while self.hist and now - self.hist[0] > self.period:
            self.hist.popleft()
        if len(self.hist) >= self.max_req:
            time.sleep(self.period - (now - self.hist[0]) + 0.1)
        if self.hist and time.monotonic() - self.hist[-1] < self.min_gap:
            time.sleep(self.min_gap - (time.monotonic() - self.hist[-1]))
        self.hist.append(time.monotonic())


def _to_utc_naive(d) -> dt.datetime:
    if isinstance(d, dt.datetime):
        return d.astimezone(dt.timezone.utc).replace(tzinfo=None) if d.tzinfo else d
    return dt.datetime(d.year, d.month, d.day)


class IBKRData:
    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 17,
                 pacer: Pacer | None = None):
        if IB is None:
            raise ImportError("pip install ib_async")
        self.host, self.port, self.client_id = host, port, client_id
        self.ib = IB()
        self.pacer = pacer or Pacer()

    def __enter__(self):
        self.ib.connect(self.host, self.port, clientId=self.client_id, readonly=True, timeout=20)
        return self

    def __exit__(self, *a):
        self.ib.disconnect()

    # ---- contrats -----------------------------------------------------------
    def stock(self, symbol: str, exchange: str = "SMART", currency: str = "USD",
              primary: str | None = None) -> "Contract":
        c = Stock(symbol, exchange, currency, primaryExchange=primary or "")
        return self.ib.qualifyContracts(c)[0]

    def index(self, symbol: str, exchange: str) -> "Contract":
        return self.ib.qualifyContracts(Index(symbol, exchange))[0]

    # ---- historique ---------------------------------------------------------
    def bars(self, contract, end: dt.datetime | str = "", duration: str = "1 M",
             bar_size: str = "1 hour", what: str = "TRADES", rth: bool = True) -> pl.DataFrame:
        self.pacer.wait()
        if isinstance(end, dt.datetime) and end.tzinfo is None:
            end = end.replace(tzinfo=dt.timezone.utc)
        bl = self.ib.reqHistoricalData(contract, endDateTime=end, durationStr=duration,
                                       barSizeSetting=bar_size, whatToShow=what, useRTH=rth,
                                       formatDate=2, timeout=120)
        if not bl:
            return pl.DataFrame()
        return pl.DataFrame({
            "ts": [_to_utc_naive(b.date) for b in bl],
            "open": [b.open for b in bl], "high": [b.high for b in bl],
            "low": [b.low for b in bl], "close": [b.close for b in bl],
            "volume": [float(b.volume) if b.volume is not None and b.volume >= 0 else None for b in bl],
            "vwap": [b.average for b in bl], "n_trades": [b.barCount for b in bl],
        }).with_columns(pl.col("ts").cast(pl.Datetime("us")))

    def history(self, contract, start: dt.datetime, end: dt.datetime | None = None,
                bar_size: str = "1 hour", what: str = "TRADES", rth: bool = True) -> pl.DataFrame:
        """Remonte le temps par blocs depuis `end` jusqu'à `start`."""
        chunks, cur = [], end or dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        while cur > start:
            df = self.bars(contract, cur, CHUNK.get(bar_size, "1 M"), bar_size, what, rth)
            if df.is_empty():
                break
            chunks.append(df)
            first = df["ts"].min()
            if first >= cur:
                break
            cur = first
        if not chunks:
            return pl.DataFrame()
        return pl.concat(chunks).unique("ts").filter(pl.col("ts") >= start).sort("ts")

    def download(self, symbols: list[str], start: dt.datetime, bar_size: str, store: ParquetStore,
                 kind: str = "stk", what: str = "TRADES", rth: bool = True, incremental: bool = True):
        """Télécharge une liste d'actions dans le store (incrémental : reprend au dernier ts)."""
        failed = []
        for i, sym in enumerate(symbols):
            try:
                s = start
                if incremental and (last := store.last_ts(kind, bar_size, sym)) is not None:
                    s = last
                df = self.history(self.stock(sym), s, bar_size=bar_size, what=what, rth=rth)
                if not df.is_empty():
                    store.write(df, kind, bar_size, sym)
                print(f"[{i + 1}/{len(symbols)}] {sym}: {len(df)} barres")
            except Exception as e:  # noqa: BLE001
                failed.append((sym, str(e)))
                print(f"[{i + 1}/{len(symbols)}] {sym}: ERREUR {e}")
        return failed

    def download_market_context(self, start: dt.datetime, bar_size: str, store: ParquetStore,
                                rth: bool = True):
        """Indices (niveau, sans volume) + ETF (avec volume) pour régimes et facteurs."""
        for sym, exch in MARKET_INDEXES:
            df = self.history(self.index(sym, exch), start, bar_size=bar_size, what="TRADES", rth=rth)
            if not df.is_empty():
                store.write(df, "idx", bar_size, sym)
        self.download(MARKET_ETFS, start, bar_size, store, kind="etf", rth=rth)


def load_market_table(store: ParquetStore, bar_size: str, proxy: str = "SPY",
                      vix: str | None = "VIX") -> pl.DataFrame:
    """Table marché (ts, mkt_ret, [vix]) depuis le store."""
    m = (store.scan("etf", bar_size, [proxy]).select("ts", "close").collect().sort("ts")
         .select("ts", mkt_ret=pl.col("close").log().diff()))
    if vix:
        v = store.scan("idx", bar_size, [vix]).select("ts", vix=pl.col("close")).collect()
        m = m.join(v, on="ts", how="left").with_columns(pl.col("vix").forward_fill())
    return m
