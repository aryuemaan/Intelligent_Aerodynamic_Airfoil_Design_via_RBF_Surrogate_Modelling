import numpy as np
from airfoil_rbf.features.aerodynamics import (
    velocity_over_ainf, surface_cp_incompressible)
from airfoil_rbf.features.style import compute_style_features
from airfoil_rbf.synthetic import naca4


def test_cp_freestream_is_zero():
    # At free-stream speed V=V_inf, Cp should be ~0.
    v_over_ainf = np.array([0.1])      # == mach_inf
    cp = surface_cp_incompressible(v_over_ainf, mach_inf=0.1)
    assert abs(cp[0]) < 1e-9


def test_cp_stagnation_is_one():
    cp = surface_cp_incompressible(np.array([0.0]), mach_inf=0.1)
    assert abs(cp[0] - 1.0) < 1e-9


def test_velocity_from_conservative():
    v = velocity_over_ainf(density=np.array([1.0]),
                           momentum_x=np.array([0.06]),
                           momentum_y=np.array([0.08]))
    assert np.isclose(v[0], 0.1)       # sqrt(0.06^2+0.08^2)=0.1


def test_style_features_finite_on_real_shape():
    lm = naca4(0.02, 0.4, 0.12, n=200)
    # fabricate a plausible suction-heavy Cp loop
    x = lm[:, 0]
    cp = -1.5 * np.exp(-((x - 0.3) ** 2) / 0.05) + 0.2
    v = compute_style_features(cp, lm)
    assert v.shape == (4,)
    assert np.all(np.isfinite(v))
