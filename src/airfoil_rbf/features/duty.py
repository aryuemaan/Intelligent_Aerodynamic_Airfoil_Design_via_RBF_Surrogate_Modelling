"""Duty (operating-condition) features.

For the OEDI 9k dataset M_inf and Re are fixed, so the only varying duty
parameter is angle of attack. We keep the interface general (a vector) so
the framework extends cleanly if you later enrich the duty space (e.g.
adding intermediate AoA via XFOIL).
"""
from __future__ import annotations

import numpy as np


def duty_vector(aoa_deg, mach_inf=None, reynolds=None):
    """Return the duty feature vector. By default just [AoA]."""
    feats = [float(aoa_deg)]
    if mach_inf is not None:
        feats.append(float(mach_inf))
    if reynolds is not None:
        feats.append(float(reynolds))
    return np.array(feats, dtype=float)
