"""Generate a small synthetic dataset in the canonical OEDI layout.

This lets you validate the ENTIRE pipeline on Windows/Git Bash in seconds,
before downloading the real 52.7 GB dataset. The airfoils are NACA-4-digit
shapes; the 'flow field' is a toy incompressible surface-speed model so that
Cp, the style features, and CL/CD are mutually consistent (good enough to
exercise every code path, not physically authoritative).

Canonical layout written here (the loader treats this as the reference):

    synthetic_9k.h5
      /shape_0000/
          landmarks                 (400, 2)
          /aoa_4/   x y density momentum_x momentum_y energy CL CD CM
          /aoa_12/  x y density momentum_x momentum_y energy CL CD CM
      /shape_0001/ ...
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import Config
from .utils import log

# NOTE: h5py is imported lazily inside generate() so that the pure-geometry
# helper naca4() (reused by the test-suite and by any caller that only needs
# shapes, not file I/O) does not require h5py to be installed.


def naca4(m, p, t, n=200):
    """NACA 4-digit airfoil as a closed loop (TE -> upper -> LE -> lower -> TE)."""
    beta = np.linspace(0.0, np.pi, n)
    x = (1.0 - np.cos(beta)) / 2.0
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                  + 0.2843 * x**3 - 0.1015 * x**4)
    yc = np.where(x < p,
                  m / max(p**2, 1e-6) * (2 * p * x - x**2),
                  m / max((1 - p)**2, 1e-6) * ((1 - 2 * p) + 2 * p * x - x**2))
    dyc = np.where(x < p,
                   2 * m / max(p**2, 1e-6) * (p - x),
                   2 * m / max((1 - p)**2, 1e-6) * (p - x))
    theta = np.arctan(dyc)
    xu, yu = x - yt * np.sin(theta), yc + yt * np.cos(theta)
    xl, yl = x + yt * np.sin(theta), yc - yt * np.cos(theta)
    upper = np.column_stack([xu, yu])[::-1]      # TE -> LE
    lower = np.column_stack([xl, yl])[1:]        # LE -> TE
    return np.vstack([upper, lower])


def _toy_flow(landmarks, aoa_deg, m, mach_inf=0.1):
    """Assign a plausible surface speed/momentum to landmark nodes + far grid."""
    x = landmarks[:, 0]
    y = landmarks[:, 1]
    suction = y >= np.interp(x, [0, 1], [0, 0])  # crude upper/lower split
    alpha = np.radians(aoa_deg)
    # speed ratio V/Vinf: a suction bump near 30% chord, stronger with camber/AoA
    bump = np.exp(-((x - 0.3) ** 2) / (2 * 0.15 ** 2))
    ratio = np.ones_like(x)
    ratio[suction] += (0.4 + 3 * m + 1.2 * alpha) * bump[suction]
    ratio[~suction] -= (0.15 + 1.0 * m) * bump[~suction]
    ratio = np.clip(ratio, 0.05, None)

    rho = np.ones_like(x)                          # rho* = 1
    speed_over_ainf = mach_inf * ratio
    # arbitrary direction; magnitude is what Cp uses
    momx = rho * speed_over_ainf * np.cos(alpha)
    momy = rho * speed_over_ainf * np.sin(alpha)
    u = momx / rho
    v = momy / rho
    energy = 1.0 / (1.4 * (1.4 - 1.0)) + 0.5 * rho * (u**2 + v**2)

    # add a coarse far-field grid (freestream) so nearest-node logic is exercised
    gx, gy = np.meshgrid(np.linspace(-0.5, 1.5, 30), np.linspace(-0.8, 0.8, 25))
    fx, fy = gx.ravel(), gy.ravel()
    f_rho = np.ones_like(fx)
    f_speed = mach_inf * np.ones_like(fx)
    f_momx = f_rho * f_speed * np.cos(alpha)
    f_momy = f_rho * f_speed * np.sin(alpha)
    f_energy = 1.0 / (1.4 * 0.4) + 0.5 * f_rho * (
        (f_momx / f_rho) ** 2 + (f_momy / f_rho) ** 2)

    node_x = np.concatenate([x, fx])
    node_y = np.concatenate([y, fy])
    density = np.concatenate([rho, f_rho])
    momentum_x = np.concatenate([momx, f_momx])
    momentum_y = np.concatenate([momy, f_momy])
    en = np.concatenate([energy, f_energy])
    return node_x, node_y, density, momentum_x, momentum_y, en


def generate(cfg: Config, n_shapes: int = 60, seed: int = 0) -> Path:
    import h5py  # lazy: only needed when actually writing the dataset file

    rng = np.random.default_rng(seed)
    out_dir = cfg.paths.raw_data
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "synthetic_9k.h5"

    with h5py.File(out_path, "w") as f:
        for sid in range(n_shapes):
            m = float(rng.uniform(0.0, 0.06))
            p = float(rng.uniform(0.3, 0.5))
            t = float(rng.uniform(0.10, 0.18))
            lm = naca4(m, p, t, n=200)
            grp = f.create_group(f"shape_{sid:04d}")
            grp.create_dataset("landmarks", data=lm.astype(np.float32))
            for aoa in cfg.flow.aoas:
                nx, ny, rho, mx, my, en = _toy_flow(lm, aoa, m, cfg.flow.mach_inf)
                a = grp.create_group(f"aoa_{int(aoa)}")
                a.create_dataset("x", data=nx.astype(np.float32))
                a.create_dataset("y", data=ny.astype(np.float32))
                a.create_dataset("density", data=rho.astype(np.float32))
                a.create_dataset("momentum_x", data=mx.astype(np.float32))
                a.create_dataset("momentum_y", data=my.astype(np.float32))
                a.create_dataset("energy", data=en.astype(np.float32))
                alpha = np.radians(aoa)
                cl = 2 * np.pi * alpha + 4.0 * m + rng.normal(0, 0.01)
                cd = 0.008 + 0.02 * alpha**2 + 0.3 * m**2 + abs(rng.normal(0, 0.001))
                cm = -0.1 * m - 0.02 * alpha + rng.normal(0, 0.002)
                a.create_dataset("CL", data=np.float32(cl))
                a.create_dataset("CD", data=np.float32(cd))
                a.create_dataset("CM", data=np.float32(cm))
    log.info("Wrote synthetic dataset (%d shapes) -> %s", n_shapes, out_path)
    return out_path
