"""Min-max scaler to [0,1] with constant-feature protection and npz I/O."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class MinMaxScaler:
    def __init__(self):
        self.min_ = None
        self.range_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.min_ = X.min(axis=0)
        rng = X.max(axis=0) - self.min_
        rng[rng == 0] = 1.0
        self.range_ = rng
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.min_) / self.range_

    def inverse_transform(self, Xs):
        return np.asarray(Xs, dtype=float) * self.range_ + self.min_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def save(self, path):
        np.savez(path, min_=self.min_, range_=self.range_)

    @classmethod
    def load(cls, path):
        d = np.load(Path(path))
        s = cls()
        s.min_, s.range_ = d["min_"], d["range_"]
        return s
