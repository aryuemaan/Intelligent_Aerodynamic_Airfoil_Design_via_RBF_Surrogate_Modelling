import numpy as np
from airfoil_rbf.models.rbf import RBFNetwork, SciPyRBF


def _data():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(120, 3))
    Y = np.column_stack([
        np.sin(3 * X[:, 0]) + X[:, 1] ** 2,
        X[:, 2] - X[:, 0] * X[:, 1],
    ])
    return X, Y


def test_custom_rbf_interpolates_training_points():
    X, Y = _data()
    m = RBFNetwork(epsilon=0.5).fit(X, Y)
    pred = m.predict(X)
    assert np.allclose(pred, Y, atol=1e-3)


def test_scipy_rbf_low_test_error():
    X, Y = _data()
    m = SciPyRBF(epsilon=0.5).fit(X[:100], Y[:100])
    rmse = m.score_rmse(X[100:], Y[100:])
    assert rmse.mean() < 0.2
