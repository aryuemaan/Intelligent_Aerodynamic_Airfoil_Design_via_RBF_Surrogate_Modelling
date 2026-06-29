"""Post-hoc quality filters (Clark applied these during data generation;
the OEDI data is pre-computed, so we apply them after the fact).

Filters:
  F1 optional positive-CL requirement
  F2 CST fit quality (drop poorly represented geometries)
  F3 outlier drag  (CD > mean + cd_sigma*std)  -> likely separation/divergence
  F4 outlier lift  (|CL - mean| > cl_sigma*std)
  F5 non-finite / out-of-range style features
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..utils import log, save_npy


def apply_filters(cfg: Config, X, Y, rmse, perf=None):
    """Return (X_clean, Y_clean, mask).

    `rmse` is the per-row CST fit RMSE (n,2). `perf` (n,3 CL/CD/CM) is read
    from Y when direction='inverse', else passed explicitly.
    """
    n = len(X)
    valid = np.ones(n, dtype=bool)
    fc = cfg.filt

    if perf is None:
        if cfg.rbf.direction == "forward":
            perf = Y                      # Y == [CL CD CM]
        else:
            perf = Y[:, -3:]              # last three cols
    cl, cd = perf[:, 0], perf[:, 1]

    if fc.require_positive_cl:
        valid &= cl > 0.0

    # F2 CST fit quality
    valid &= np.nanmax(rmse, axis=1) <= cfg.cst.rmse_thresh

    # F3 outlier drag (compute stats on currently-valid subset)
    if valid.any():
        mu, sd = cd[valid].mean(), cd[valid].std()
        valid &= cd < (mu + fc.cd_sigma * sd)

    # F4 outlier lift
    if valid.any():
        mu, sd = cl[valid].mean(), cl[valid].std()
        valid &= np.abs(cl - mu) < (fc.cl_sigma * sd)

    # F5 finite features
    valid &= np.all(np.isfinite(X), axis=1)
    valid &= np.all(np.isfinite(Y), axis=1)
    # bound style columns when present (inverse direction: cols 1..)
    if cfg.rbf.direction == "inverse" and X.shape[1] > 1:
        valid &= np.all(np.abs(X[:, 1:]) < fc.style_abs_max, axis=1)

    Xc, Yc = X[valid], Y[valid]
    log.info("Filter kept %d / %d rows (%.1f%%)",
             valid.sum(), n, 100.0 * valid.sum() / max(n, 1))

    save_npy(cfg.paths.work / "X_clean.npy", Xc)
    save_npy(cfg.paths.work / "Y_clean.npy", Yc)
    save_npy(cfg.paths.work / "valid_mask.npy", valid)
    return Xc, Yc, valid
