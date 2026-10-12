"""Walk-forward purgé + embargo (López de Prado).

On découpe sur les DATES uniques (pas les lignes) : toutes les actions d'une
même date sont dans le même pli. Entre train et test on retire
gap = lag + horizon + embargo dates, car la cible y_h d'une date t chevauche
[t+lag, t+lag+h] : sans purge, le train "voit" le début du test.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class PurgedWalkForward:
    min_train: int = 500          # dates minimum avant le 1er test
    test_size: int = 60           # dates par pli de test (= fréquence de ré-entraînement)
    horizon: int = 1
    lag: int = 1
    embargo: int = 5
    train_window: int | None = None   # None = fenêtre expansive

    @property
    def gap(self) -> int:
        return self.lag + self.horizon + self.embargo

    def split(self, n_dates: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        k = self.min_train
        while k < n_dates:
            te = np.arange(k, min(k + self.test_size, n_dates))
            tr_end = k - self.gap
            tr_start = 0 if self.train_window is None else max(0, tr_end - self.train_window)
            if tr_end - tr_start > 50:
                yield np.arange(tr_start, tr_end), te
            k += self.test_size

    def inner_split(self, train_idx: np.ndarray, val_frac: float = 0.25):
        """Découpe interne (sélection d'hyperparamètres) : la fin du train sert de validation."""
        n = len(train_idx)
        cut = int(n * (1 - val_frac))
        return train_idx[: max(cut - self.gap, 1)], train_idx[cut:]
