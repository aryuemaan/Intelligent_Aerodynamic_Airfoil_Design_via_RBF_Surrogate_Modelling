"""Recover surface pressure coefficient from the NREL conservative-variable
flow field.

The dataset stores per-node density, x/y momentum, total energy and
vorticity (Mach-normalised: momentum magnitude at free-stream = M_inf).
Surface Cp is not stored, so we compute it.

Two routes, both selectable from config (flow.cp_method):

  incompressible (default, robust at M=0.1):
      V/a_inf      = |momentum| / density           (local speed / a_inf)
      V/V_inf      = (V/a_inf) / M_inf
      Cp           = 1 - (V/V_inf)^2                 (Bernoulli, exact for
                                                      incompressible flow;
                                                      <1% error at M=0.1)

  compressible (uses energy; needs the file's exact energy normalisation):
      u, v   = momentum / density
      p      = (gamma-1) * (E - 0.5*rho*(u^2+v^2))
      Cp     = (p - p_inf) / (0.5 * rho_inf * V_inf^2)

Surface values are obtained by nearest-node lookup from the airfoil
landmarks into the mesh (KD-tree), so we never need to load or keep the
full volume field beyond the read.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def velocity_over_ainf(density, momentum_x, momentum_y, eps=1e-12):
    """Local speed normalised by free-stream speed of sound, |V|/a_inf.

    Works directly on the dataset's Mach-normalised conservative variables
    because (rho* u / a_inf) / rho* = u / a_inf.
    """
    density = np.asarray(density, dtype=float)
    mx = np.asarray(momentum_x, dtype=float)
    my = np.asarray(momentum_y, dtype=float)
    denom = np.where(np.abs(density) < eps, eps, density)
    u = mx / denom
    v = my / denom
    return np.sqrt(u * u + v * v)


def surface_cp_incompressible(v_over_ainf, mach_inf=0.1):
    """Cp = 1 - (V/V_inf)^2 with V_inf = mach_inf * a_inf."""
    v_over_vinf = np.asarray(v_over_ainf, dtype=float) / max(mach_inf, 1e-9)
    return 1.0 - v_over_vinf ** 2


def surface_cp_compressible(density, momentum_x, momentum_y, energy,
                            gamma=1.4, mach_inf=0.1, rho_inf=1.0):
    """Cp from full conservative variables (see module docstring).

    Densities/energies are taken in the file's own normalised units; the
    free-stream reference state is reconstructed self-consistently from
    mach_inf so the result is dimensionless.
    """
    density = np.asarray(density, dtype=float)
    mx = np.asarray(momentum_x, dtype=float)
    my = np.asarray(momentum_y, dtype=float)
    energy = np.asarray(energy, dtype=float)

    denom = np.where(np.abs(density) < 1e-12, 1e-12, density)
    u = mx / denom
    v = my / denom
    p = (gamma - 1.0) * (energy - 0.5 * density * (u * u + v * v))

    # Free-stream reference (normalised): rho=rho_inf, speed = mach_inf
    p_inf = rho_inf / gamma            # p_inf = rho_inf a_inf^2 / gamma (a_inf=1 units)
    q_inf = 0.5 * rho_inf * mach_inf ** 2
    return (p - p_inf) / max(q_inf, 1e-12)


def sample_surface_field(landmarks, node_xy, *fields):
    """Map node-based fields onto the airfoil surface via nearest neighbour.

    Parameters
    ----------
    landmarks : (Ns, 2) ordered airfoil surface coordinates
    node_xy   : (Nn, 2) mesh node coordinates
    *fields   : one or more (Nn,) node arrays to sample

    Returns
    -------
    sampled : list of (Ns,) arrays, one per input field, ordered to match
              the landmark ordering (LE..TE around the loop).
    """
    landmarks = np.asarray(landmarks, dtype=float)
    node_xy = np.asarray(node_xy, dtype=float)
    tree = cKDTree(node_xy)
    _, idx = tree.query(landmarks, k=1)
    return [np.asarray(f, dtype=float)[idx] for f in fields]
