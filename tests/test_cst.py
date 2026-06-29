import numpy as np
from airfoil_rbf.synthetic import naca4
from airfoil_rbf.geometry import fit_airfoil, reconstruct_airfoil


def test_cst_roundtrip_low_error():
    lm = naca4(m=0.02, p=0.4, t=0.12, n=200)
    geom, yte, rmse = fit_airfoil(lm, n_order=12, n_surface_pts=120)
    # CST should represent a smooth NACA shape well
    assert rmse.max() < 1e-3
    assert geom.shape == (2 * (12 + 1),)


def test_reconstruct_shapes_consistent():
    lm = naca4(m=0.04, p=0.4, t=0.15, n=160)
    geom, yte, _ = fit_airfoil(lm, n_order=10)
    psi, y_u, y_l = reconstruct_airfoil(geom, y_te=yte)
    assert psi.shape == y_u.shape == y_l.shape
    # upper surface should sit above lower for a normal airfoil
    assert np.mean(y_u) > np.mean(y_l)
