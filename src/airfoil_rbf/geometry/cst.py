"""Class-Shape Transformation (CST / Kulfan) parameterisation.

    zeta(psi) = C(psi) * S(psi) + psi * zeta_TE
    C(psi)    = psi^N1 (1-psi)^N2                  (class function)
    S(psi)    = sum_i A_i * K_{i,N} psi^i (1-psi)^{N-i}   (Bernstein shape)

Each airfoil is described by upper + lower coefficient vectors, giving a
compact fixed-length geometry vector of length 2*(n_order+1).
"""
from __future__ import annotations

import numpy as np
from scipy.special import comb

from .surface import split_surfaces, resample_surface


def bernstein_basis(psi: np.ndarray, n_order: int) -> np.ndarray:
    """(len(psi), n_order+1) Bernstein basis matrix."""
    psi = np.asarray(psi, dtype=float)
    B = np.empty((psi.size, n_order + 1))
    for i in range(n_order + 1):
        k = comb(n_order, i, exact=True)
        B[:, i] = k * psi**i * (1.0 - psi) ** (n_order - i)
    return B


def class_function(psi: np.ndarray, n1: float = 0.5, n2: float = 1.0) -> np.ndarray:
    psi = np.asarray(psi, dtype=float)
    return psi**n1 * (1.0 - psi) ** n2


def cst_design_matrix(psi, n_order, n1=0.5, n2=1.0) -> np.ndarray:
    """Design matrix D with D[:,i] = C(psi) * B_{i,N}(psi)."""
    C = class_function(psi, n1, n2)
    B = bernstein_basis(psi, n_order)
    return C[:, None] * B


def fit_cst(surface, n_order=10, n1=0.5, n2=1.0):
    """Least-squares CST fit to one (M,2) surface.

    Returns (coeffs[n_order+1], y_te, rmse).
    The trailing-edge ordinate is removed as a linear ramp before fitting
    so the Bernstein series models only the camber/thickness shape.
    """
    surface = np.asarray(surface, dtype=float)
    x, y = surface[:, 0], surface[:, 1]
    y_te = float(y[-1])

    D = cst_design_matrix(x, n_order, n1, n2)
    y_adj = y - x * y_te
    coeffs, *_ = np.linalg.lstsq(D, y_adj, rcond=None)

    y_pred = D @ coeffs + x * y_te
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    return coeffs, y_te, rmse


def reconstruct_cst(coeffs, psi, y_te=0.0, n1=0.5, n2=1.0) -> np.ndarray:
    """Reconstruct surface y/c from CST coefficients."""
    n_order = len(coeffs) - 1
    D = cst_design_matrix(psi, n_order, n1, n2)
    return D @ coeffs + np.asarray(psi, dtype=float) * y_te


def fit_airfoil(xy, n_order=10, n_surface_pts=101, n1=0.5, n2=1.0):
    """Fit CST to a full airfoil given its closed landmark loop.

    Returns
    -------
    geom_vec : (2*(n_order+1),) concatenated [upper coeffs, lower coeffs]
    y_te     : (2,) trailing-edge ordinates [upper, lower]
    rmse     : (2,) per-surface fit RMSE [upper, lower]
    """
    upper, lower = split_surfaces(xy)
    upper = resample_surface(upper, n_surface_pts)
    lower = resample_surface(lower, n_surface_pts)

    cu, yte_u, ru = fit_cst(upper, n_order, n1, n2)
    cl, yte_l, rl = fit_cst(lower, n_order, n1, n2)

    geom_vec = np.concatenate([cu, cl])
    return geom_vec, np.array([yte_u, yte_l]), np.array([ru, rl])


def reconstruct_airfoil(geom_vec, psi=None, y_te=(0.0, 0.0), n1=0.5, n2=1.0):
    """Reconstruct upper/lower surfaces from a concatenated geometry vector.

    Returns (psi, y_upper, y_lower).
    """
    if psi is None:
        beta = np.linspace(0.0, np.pi, 201)
        psi = (1.0 - np.cos(beta)) / 2.0
    geom_vec = np.asarray(geom_vec, dtype=float)
    n = geom_vec.size // 2
    cu, cl = geom_vec[:n], geom_vec[n:]
    y_u = reconstruct_cst(cu, psi, y_te[0], n1, n2)
    y_l = reconstruct_cst(cl, psi, y_te[1], n1, n2)
    return psi, y_u, y_l
