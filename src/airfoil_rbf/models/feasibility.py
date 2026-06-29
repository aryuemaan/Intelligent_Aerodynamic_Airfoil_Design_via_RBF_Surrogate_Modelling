"""Infeasible-region detection (Clark's idea): a query in (duty, style) space
is 'infeasible' if the surrogate's local reconstruction error is high, i.e.
no physically realisable airfoil corresponds to that demand.

We fit a secondary RBF that maps inputs -> geometric reconstruction error,
and flag queries whose predicted error exceeds a percentile threshold.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RBFInterpolator


class FeasibilityModel:
    def __init__(self, percentile=90.0, epsilon=1.0, smoothing=1e-4):
        self.percentile = percentile
        self.epsilon = epsilon
        self.smoothing = smoothing
        self.threshold = None
        self._rbf = None

    def fit(self, X_norm, geo_errors):
        geo_errors = np.asarray(geo_errors, float).reshape(-1)
        self.threshold = float(np.percentile(geo_errors, self.percentile))
        self._rbf = RBFInterpolator(
            np.asarray(X_norm, float), geo_errors[:, None],
            kernel="multiquadric", epsilon=self.epsilon,
            smoothing=self.smoothing)
        return self

    def predict_error(self, X_norm):
        X_norm = np.atleast_2d(np.asarray(X_norm, float))
        return self._rbf(X_norm).reshape(-1)

    def is_feasible(self, x_norm):
        return bool(self.predict_error(x_norm)[0] < self.threshold)
