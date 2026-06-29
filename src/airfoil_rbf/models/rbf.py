"""Radial Basis Function surrogates.

Two implementations:
  RBFNetwork - transparent multiquadric RBF following Clark (GT2019-91637):
               phi(r)=sqrt(r^2+eps^2), weights from a least-squares solve of
               Phi W = Y. Educational and fully inspectable.
  SciPyRBF   - thin wrapper over scipy.interpolate.RBFInterpolator, which is
               numerically robust and supports neighbour-limited solves for
               large N (the practical choice for the full 9k dataset).

`build_rbf` picks centres (optionally a capped random subset) and returns a
fitted model plus the centre indices used.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import RBFInterpolator


class RBFNetwork:
    """Multiquadric RBF with one centre per training point."""

    def __init__(self, epsilon=1.0):
        self.epsilon = float(epsilon)
        self.centres = None
        self.weights = None

    def _kernel(self, r):
        return np.sqrt(r ** 2 + self.epsilon ** 2)

    def fit(self, X, Y):
        X = np.asarray(X, dtype=float)
        self.centres = X.copy()
        Phi = self._kernel(cdist(X, self.centres))
        self.weights, *_ = np.linalg.lstsq(Phi, np.asarray(Y, float), rcond=None)
        return self

    def predict(self, Xq):
        Phi = self._kernel(cdist(np.asarray(Xq, float), self.centres))
        return Phi @ self.weights

    def score_rmse(self, X, Y):
        return np.sqrt(np.mean((self.predict(X) - np.asarray(Y, float)) ** 2,
                               axis=0))


class SciPyRBF:
    """Wrapper providing the same predict/score API as RBFNetwork."""

    def __init__(self, epsilon=1.0, kernel="multiquadric", smoothing=0.0,
                 neighbors=None):
        self.epsilon = float(epsilon)
        self.kernel = kernel
        self.smoothing = float(smoothing)
        self.neighbors = neighbors
        self._rbf = None

    def fit(self, X, Y):
        self._rbf = RBFInterpolator(
            np.asarray(X, float), np.asarray(Y, float),
            kernel=self.kernel, epsilon=self.epsilon,
            smoothing=self.smoothing, neighbors=self.neighbors)
        return self

    def predict(self, Xq):
        return self._rbf(np.asarray(Xq, float))

    def score_rmse(self, X, Y):
        return np.sqrt(np.mean((self.predict(X) - np.asarray(Y, float)) ** 2,
                               axis=0))


def build_rbf(X, Y, epsilon, cfg_rbf, backend="scipy"):
    """Fit an RBF, optionally on a capped random subset of centres.

    Returns (model, centre_idx). With RBFInterpolator we pass the subset as
    the full interpolation set (exact at those points); this is the standard
    approximation when N is too large for a dense N x N solve.
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    n = len(X)
    cap = cfg_rbf.max_centres
    if cap and n > cap:
        rng = np.random.default_rng(cfg_rbf.random_state)
        idx = rng.choice(n, size=cap, replace=False)
    else:
        idx = np.arange(n)

    Xc, Yc = X[idx], Y[idx]
    if backend == "custom":
        model = RBFNetwork(epsilon=epsilon).fit(Xc, Yc)
    else:
        model = SciPyRBF(epsilon=epsilon, kernel=cfg_rbf.kernel,
                         smoothing=cfg_rbf.smoothing).fit(Xc, Yc)
    return model, idx


def save_rbf(model, path):
    """Persist a fitted SciPyRBF/RBFNetwork by storing centres + outputs.

    We re-store the training (centre, value) pairs and refit on load, which
    is robust across scipy versions.
    """
    path = Path(path)
    if isinstance(model, SciPyRBF):
        rbf = model._rbf
        np.savez(path, backend="scipy", epsilon=model.epsilon,
                 kernel=model.kernel, smoothing=model.smoothing,
                 X=rbf.y, Y=rbf.d)
    else:
        np.savez(path, backend="custom", epsilon=model.epsilon,
                 X=model.centres, Y=model.weights, kernel="multiquadric",
                 smoothing=0.0)


def load_rbf(path):
    d = np.load(Path(path), allow_pickle=False)
    backend = str(d["backend"])
    eps = float(d["epsilon"])
    if backend == "scipy":
        m = SciPyRBF(epsilon=eps, kernel=str(d["kernel"]),
                     smoothing=float(d["smoothing"]))
        return m.fit(d["X"], d["Y"])
    m = RBFNetwork(epsilon=eps)
    m.centres = d["X"]
    m.weights = d["Y"]
    return m
