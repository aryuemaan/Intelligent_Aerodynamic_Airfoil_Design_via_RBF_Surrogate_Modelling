"""Train / validate the RBF surrogate and evaluate it.

Splits data, cross-validates the multiquadric shape parameter epsilon,
fits the final model, and reports geometry + performance accuracy.
Saves the model, scalers, and a small metrics JSON to the models dir.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from .config import Config
from .models.scaler import MinMaxScaler
from .models.rbf import build_rbf, save_rbf, SciPyRBF
from .models.feasibility import FeasibilityModel
from .utils import log, timer


def _split(n, cfg: Config):
    idx = np.arange(n)
    rs, ts, vs = cfg.rbf.random_state, cfg.rbf.test_size, cfg.rbf.val_size
    tr, tmp = train_test_split(idx, test_size=ts + vs, random_state=rs)
    rel = ts / (ts + vs)
    val, test = train_test_split(tmp, test_size=rel, random_state=rs)
    return tr, val, test


def _eval_eps(eps, Xtr, Ytr, Xva, Yva, cfg):
    model, _ = build_rbf(Xtr, Ytr, eps, cfg.rbf)
    return eps, float(model.score_rmse(Xva, Yva).mean())


def train(cfg: Config):
    work, mdir = cfg.paths.work, cfg.paths.models
    X = np.load(work / "X_clean.npy")
    Y = np.load(work / "Y_clean.npy")
    log.info("Training on X%s Y%s (direction=%s)", X.shape, Y.shape,
             cfg.rbf.direction)

    xs = MinMaxScaler().fit(X)
    ys = MinMaxScaler().fit(Y)
    Xn, Yn = xs.transform(X), ys.transform(Y)

    tr, val, test = _split(len(Xn), cfg)
    Xtr, Ytr = Xn[tr], Yn[tr]
    Xva, Yva = Xn[val], Yn[val]
    Xte, Yte = Xn[test], Yn[test]

    # ---- epsilon cross-validation ----
    with timer("epsilon CV"):
        results = Parallel(n_jobs=cfg.n_jobs)(
            delayed(_eval_eps)(e, Xtr, Ytr, Xva, Yva, cfg)
            for e in cfg.rbf.epsilons)
    eps_vals, rmses = zip(*results)
    best_eps = float(eps_vals[int(np.argmin(rmses))])
    log.info("Best epsilon = %g (val RMSE %.5f)", best_eps, min(rmses))

    # ---- final fit ----
    t0 = time.time()
    model, centre_idx = build_rbf(Xtr, Ytr, best_eps, cfg.rbf)
    fit_time = time.time() - t0
    test_rmse = model.score_rmse(Xte, Yte)
    log.info("Final fit: %d centres, %.1fs, test RMSE (mean) %.5f",
             len(centre_idx), fit_time, float(test_rmse.mean()))

    # ---- feasibility model (inverse direction only) ----
    feas_path = None
    if cfg.rbf.direction == "inverse":
        g = Y.shape[1] - 3
        Yhat_all = ys.inverse_transform(model.predict(Xn))
        geo_err = np.sqrt(np.mean((Yhat_all[:, :g] - Y[:, :g]) ** 2, axis=1))
        feas = FeasibilityModel(epsilon=best_eps).fit(Xn, geo_err)
        feas_path = mdir / "feasibility.npz"
        np.savez(feas_path, X=Xn, err=geo_err, threshold=feas.threshold,
                 epsilon=best_eps)

    # ---- persist ----
    save_rbf(model, mdir / "rbf_model.npz")
    xs.save(mdir / "x_scaler.npz")
    ys.save(mdir / "y_scaler.npz")
    np.savez(mdir / "split.npz", train=tr, val=val, test=test)

    metrics = {
        "direction": cfg.rbf.direction,
        "best_epsilon": best_eps,
        "n_train": int(len(tr)), "n_val": int(len(val)),
        "n_test": int(len(test)), "n_centres": int(len(centre_idx)),
        "fit_time_s": fit_time,
        "test_rmse_mean": float(test_rmse.mean()),
        "test_rmse_per_output": test_rmse.tolist(),
    }
    with open(mdir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    log.info("Saved model + scalers + metrics to %s", mdir)
    return metrics


def evaluate(cfg: Config):
    """Reload the model and produce parity plots + a metrics summary."""
    from .models.rbf import load_rbf
    from .viz import plot_parity

    work, mdir, fdir = cfg.paths.work, cfg.paths.models, cfg.paths.figures
    X = np.load(work / "X_clean.npy")
    Y = np.load(work / "Y_clean.npy")
    xs = MinMaxScaler.load(mdir / "x_scaler.npz")
    ys = MinMaxScaler.load(mdir / "y_scaler.npz")
    model = load_rbf(mdir / "rbf_model.npz")
    test = np.load(mdir / "split.npz")["test"]

    Xn = xs.transform(X)
    Yte_true = Y[test]
    Yte_pred = ys.inverse_transform(model.predict(Xn[test]))

    if cfg.rbf.direction == "forward":
        labels = ["CL", "CD", "CM"]
        plot_parity(Yte_true, Yte_pred, labels, fdir / "perf_parity.png")
        perf_true, perf_pred = Yte_true, Yte_pred
    else:
        g = Y.shape[1] - 3
        perf_true, perf_pred = Yte_true[:, g:], Yte_pred[:, g:]
        plot_parity(perf_true, perf_pred, ["CL", "CD", "CM"],
                    fdir / "perf_parity.png")
        geo_rmse = np.sqrt(np.mean((Yte_pred[:, :g] - Yte_true[:, :g]) ** 2))
        log.info("Geometry RMSE (CST space): %.6f", geo_rmse)

    for j, lab in enumerate(["CL", "CD", "CM"]):
        t, p = perf_true[:, j], perf_pred[:, j]
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        denom = np.where(np.abs(t) < 1e-9, np.nan, t)
        rel = float(np.nanmean(np.abs((p - t) / denom)) * 100)
        log.info("%s  RMSE=%.5f  mean-rel-err=%.2f%%", lab, rmse, rel)

    log.info("Saved parity plots to %s", fdir)
