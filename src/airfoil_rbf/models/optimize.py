"""Use the trained surrogate for actual design tasks.

design_from_style : inverse map. Given AoA + 4 style scalars, predict the
                    CST geometry and the CL/CD/CM it should achieve.

optimize_airfoil  : couple the FORWARD surrogate ([geometry, AoA]->CL,CD,CM)
                    with scipy.optimize to search geometry space for a target
                    objective (e.g. maximise L/D, or hit a target CL). This is
                    the end goal: intelligent, near-instant aerodynamic design.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import differential_evolution

from ..geometry import reconstruct_airfoil


def design_from_style(aoa_deg, style_vec, model, x_scaler, y_scaler,
                      n_order, feasibility=None):
    """Inverse design: predict geometry + performance from a style demand."""
    x_raw = np.concatenate([[float(aoa_deg)], np.asarray(style_vec, float)])
    x_n = x_scaler.transform(x_raw[None, :])

    warn = None
    if feasibility is not None and not feasibility.is_feasible(x_n[0]):
        warn = "Query lies in a sparse/infeasible region; result is extrapolated."

    y = y_scaler.inverse_transform(model.predict(x_n))[0]
    g = 2 * (n_order + 1)
    geom = y[:g]
    cl, cd, cm = y[g], y[g + 1], y[g + 2]
    return {"geometry": geom, "CL": float(cl), "CD": float(cd),
            "CM": float(cm), "warning": warn}


def _forward_perf(geom_vec, aoa_deg, model, x_scaler, y_scaler):
    """Evaluate the forward surrogate: geometry + AoA -> CL, CD, CM."""
    x_raw = np.concatenate([np.asarray(geom_vec, float), [float(aoa_deg)]])
    x_n = x_scaler.transform(x_raw[None, :])
    y = y_scaler.inverse_transform(model.predict(x_n))[0]
    return y[0], y[1], y[2]   # CL, CD, CM


def optimize_airfoil(forward_model, x_scaler, y_scaler, geom_bounds,
                     aoa_deg=4.0, objective="ld", target_cl=None,
                     n_order=10, maxiter=60, seed=42):
    """Optimise CST geometry against the forward surrogate.

    Parameters
    ----------
    forward_model : surrogate trained with direction='forward'
    geom_bounds   : (G, 2) per-coefficient (low, high) search bounds; derive
                    these from the training geometry min/max for realism.
    objective     : 'ld'      -> maximise CL/CD
                    'cl'       -> maximise CL
                    'target_cl'-> hit target_cl with minimum CD
    Returns a dict with the optimal geometry and predicted performance.
    """
    geom_bounds = np.asarray(geom_bounds, float)

    def cost(geom):
        cl, cd, cm = _forward_perf(geom, aoa_deg, forward_model,
                                   x_scaler, y_scaler)
        cd = max(cd, 1e-6)
        if objective == "cl":
            return -cl
        if objective == "target_cl" and target_cl is not None:
            return abs(cl - target_cl) + 0.1 * cd
        return -cl / cd  # default: maximise lift-to-drag

    result = differential_evolution(
        cost, bounds=list(map(tuple, geom_bounds)),
        maxiter=maxiter, tol=1e-6, seed=seed, polish=True, updating="deferred")

    geom = result.x
    cl, cd, cm = _forward_perf(geom, aoa_deg, forward_model, x_scaler, y_scaler)
    psi, y_u, y_l = reconstruct_airfoil(geom)
    return {"geometry": geom, "CL": float(cl), "CD": float(cd),
            "CM": float(cm), "LD": float(cl / max(cd, 1e-6)),
            "psi": psi, "y_upper": y_u, "y_lower": y_l,
            "objective_value": float(result.fun)}
