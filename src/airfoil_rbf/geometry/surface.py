"""Split a closed airfoil landmark loop into upper/lower surfaces and
resample each onto a common chordwise grid.

The NREL 'landmarks' array gives 400 (x, y) points around the airfoil.
We do NOT assume a particular winding direction; we detect the leading
edge (min x) and orient both surfaces from leading edge (x=0) to
trailing edge (x=1), then normalise the chord to [0, 1].
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


def _normalise_chord(xy: np.ndarray) -> np.ndarray:
    """Translate LE to x=0 and scale so TE is at x=1 (chord-normalised)."""
    x = xy[:, 0]
    x_min, x_max = x.min(), x.max()
    chord = x_max - x_min
    if chord <= 0:
        return xy.copy()
    out = xy.copy().astype(float)
    out[:, 0] = (xy[:, 0] - x_min) / chord
    out[:, 1] = xy[:, 1] / chord
    return out


def split_surfaces(xy: np.ndarray):
    """Split a closed (N, 2) loop into upper and lower surfaces.

    Returns (upper, lower), each an (M, 2) array ordered LE -> TE.
    """
    xy = _normalise_chord(np.asarray(xy, dtype=float))
    le_idx = int(np.argmin(xy[:, 0]))

    # Two arcs from the LE around to the loop endpoints.
    arc_a = xy[: le_idx + 1][::-1]   # LE -> start of array
    arc_b = xy[le_idx:]              # LE -> end of array

    # The arc whose mean y is larger is the upper (suction) surface.
    upper, lower = (arc_a, arc_b) if arc_a[:, 1].mean() >= arc_b[:, 1].mean() \
        else (arc_b, arc_a)
    return ensure_le_to_te(upper), ensure_le_to_te(lower)


def ensure_le_to_te(surface: np.ndarray) -> np.ndarray:
    """Order a surface so x increases (leading edge first)."""
    surface = np.asarray(surface, dtype=float)
    if surface[0, 0] > surface[-1, 0]:
        surface = surface[::-1]
    return surface


def resample_surface(surface: np.ndarray, n_pts: int = 101) -> np.ndarray:
    """Resample a surface onto n_pts using a cosine x-distribution.

    Cosine clustering concentrates points near the leading and trailing
    edges where curvature is highest, matching how the dataset itself
    samples landmarks and improving the CST least-squares fit.
    """
    surface = ensure_le_to_te(surface)
    x_src, y_src = surface[:, 0], surface[:, 1]

    # Deduplicate / sort on x for a monotone interpolant.
    order = np.argsort(x_src, kind="stable")
    x_src, y_src = x_src[order], y_src[order]
    x_src, uniq = np.unique(x_src, return_index=True)
    y_src = y_src[uniq]

    f = interp1d(x_src, y_src, kind="linear", fill_value="extrapolate")
    beta = np.linspace(0.0, np.pi, n_pts)
    x_new = (1.0 - np.cos(beta)) / 2.0            # cosine spacing in [0,1]
    x_new = x_src[0] + x_new * (x_src[-1] - x_src[0])
    return np.column_stack([x_new, f(x_new)])
