"""Full-pipeline smoke test on a tiny synthetic dataset (inverse + forward)."""
import numpy as np
import pytest

from airfoil_rbf.config import load_config
from airfoil_rbf.synthetic import generate
from airfoil_rbf.data.build_dataset import build_features, assemble_xy
from airfoil_rbf.data.filters import apply_filters
from airfoil_rbf import training


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    c = load_config()
    # redirect all outputs into the pytest tmp dir
    c.paths.raw_data = tmp_path / "raw"
    c.paths.work = tmp_path / "work"
    c.paths.models = tmp_path / "models"
    c.paths.figures = tmp_path / "figures"
    c.paths.make_dirs()
    c.rbf.max_centres = None
    c.rbf.epsilons = (0.3, 1.0)
    c.checkpoint_every = 100000  # avoid checkpoint writes in test
    generate(c, n_shapes=40, seed=1)
    return c


def test_pipeline_inverse(cfg):
    cfg.rbf.direction = "inverse"
    build_features(cfg, resume=False)
    X, Y, rmse = assemble_xy(cfg)
    assert X.shape[1] == 5            # AoA + 4 style
    Xc, Yc, mask = apply_filters(cfg, X, Y, rmse)
    assert len(Xc) > 10
    m = training.train(cfg)
    assert m["test_rmse_mean"] < 1.0
    training.evaluate(cfg)


def test_pipeline_forward(cfg):
    cfg.rbf.direction = "forward"
    build_features(cfg, resume=False)
    X, Y, rmse = assemble_xy(cfg)
    assert Y.shape[1] == 3            # CL CD CM
    Xc, Yc, mask = apply_filters(cfg, X, Y, rmse)
    training.train(cfg)
    training.evaluate(cfg)
