"""Aerodynamic-style scalars from a surface Cp distribution.

Clark (2019) parameterises 'style' via a normalised isentropic-Mach-fraction
loading curve. At M=0.1 we use the equivalent Cp loading curve and extract
four scalars (Algorithm 2 in the implementation guide):

    v1 = peak suction          = min normalised Cp on the suction side
    v2 = peak-suction location = chord fraction where v1 occurs
    v3 = LE suction level      = mean normalised Cp over x < 0.05 (suction)
    v4 = PS mid-chord level    = mean normalised Cp over 0.4 < x < 0.6 (pressure)
"""
from __future__ import annotations

import numpy as np


def _split_loop(cp, surface_xy):
    """Split a closed-loop Cp/coords array into suction (upper) and
    pressure (lower) branches, each ordered LE -> TE.

    The suction side is identified as the branch with the lower mean Cp
    (more negative) for a lifting airfoil.
    """
    cp = np.asarray(cp, dtype=float)
    xy = np.asarray(surface_xy, dtype=float)
    le_idx = int(np.argmin(xy[:, 0]))

    cp_a, x_a = cp[: le_idx + 1][::-1], xy[: le_idx + 1, 0][::-1]
    cp_b, x_b = cp[le_idx:], xy[le_idx:, 0]

    # ensure LE->TE
    if x_a[0] > x_a[-1]:
        cp_a, x_a = cp_a[::-1], x_a[::-1]
    if x_b[0] > x_b[-1]:
        cp_b, x_b = cp_b[::-1], x_b[::-1]

    if np.nanmean(cp_a) <= np.nanmean(cp_b):
        return (cp_a, x_a), (cp_b, x_b)        # (suction, pressure)
    return (cp_b, x_b), (cp_a, x_a)


def compute_style_features(cp, surface_xy):
    """Return the 4-vector [v1, v2, v3, v4]. NaNs are passed through so the
    quality filter can drop bad shapes downstream."""
    (cp_s, x_s), (cp_p, x_p) = _split_loop(cp, surface_xy)

    # Normalise by the trailing-edge Cp magnitude (loading curve).
    # Robust normalization using the Cp range instead of trailing-edge Cp
    cp_all = np.concatenate([cp_s, cp_p])

    cp_min = np.nanmin(cp_all)
    cp_max = np.nanmax(cp_all)

    scale = cp_max - cp_min

    if (not np.isfinite(scale)) or scale < 1e-6:
        scale = 1.0

    cps = (cp_s - cp_min) / scale
    cpp = (cp_p - cp_min) / scale

    # v1 / v2: peak suction value and its chordwise location.
    if cps.size and np.isfinite(cps).any():
        peak = int(np.nanargmin(cps))
        v1, v2 = float(cps[peak]), float(x_s[peak])
    else:
        v1 = v2 = np.nan

    # v3: leading-edge suction level.
    le_mask = x_s < 0.05
    v3 = float(np.nanmean(cps[le_mask])) if le_mask.any() else \
        (float(cps[0]) if cps.size else np.nan)

    # v4: pressure-side mid-chord level.
    ps_mask = (x_p > 0.4) & (x_p < 0.6)
    v4 = float(np.nanmean(cpp[ps_mask])) if ps_mask.any() else \
        (float(cpp[cpp.size // 2]) if cpp.size else np.nan)

    return np.array([v1, v2, v3, v4], dtype=float)
