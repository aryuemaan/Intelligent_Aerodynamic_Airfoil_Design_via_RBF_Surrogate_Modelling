"""Stream the (possibly 55 GB) dataset once and distil it into compact,
in-memory-friendly feature/target arrays.

For every (shape, AoA) record we produce a single row:
    aoa | style(4) | CL CD CM | CST-geometry(2*(n_order+1)) | fit-RMSE(2)

The heavy volume flow field is read, reduced to a 400-point surface Cp,
and immediately discarded, so peak memory stays small regardless of total
dataset size. Progress is checkpointed so an interrupted run resumes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # tqdm is purely cosmetic; degrade to a no-op pass-through
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else iter(())

from ..config import Config
from ..geometry import fit_airfoil
from ..features.aerodynamics import (
    velocity_over_ainf, surface_cp_incompressible, surface_cp_compressible,
    sample_surface_field,
)
from ..features.style import compute_style_features
from ..utils import save_npy, log
# NOTE: iter_records (and its h5py dependency) is imported lazily inside
# build_features() so that the pure-numpy assemble_xy() below can be used
# without h5py installed.


def _surface_cp(rec, cfg: Config):
    """Compute the surface Cp distribution for a record, or None."""
    if not rec.has_flow:
        return None
    # Sample conservative variables onto the landmark surface.
    if cfg.flow.cp_method == "compressible" and rec.energy is not None:
        rho, mx, my, E = sample_surface_field(
            rec.landmarks, rec.node_xy,
            rec.density, rec.momentum_x, rec.momentum_y, rec.energy)
        return surface_cp_compressible(
            rho, mx, my, E, gamma=cfg.flow.gamma,
            mach_inf=cfg.flow.mach_inf, rho_inf=1.0)
    rho, mx, my = sample_surface_field(
        rec.landmarks, rec.node_xy,
        rec.density, rec.momentum_x, rec.momentum_y)
    v = velocity_over_ainf(rho, mx, my)
    return surface_cp_incompressible(v, mach_inf=cfg.flow.mach_inf)


def _get_cst_coefficients(f, cfg: Config):
    """Extract CST coefficients directly from the HDF5 file if available."""
    try:
        if "shape" in f and "cst" in f["shape"]:
            cst_data = np.asarray(f["shape"]["cst"])
            return cst_data
    except Exception:
        pass
    return None


def build_features(cfg: Config, max_shapes: Optional[int] = None,
                   resume: bool = True) -> dict:
    """Run the streaming extraction. Returns a dict of stacked arrays and
    also writes them to the work directory."""
    from ..io_oedi import iter_records, find_h5_files  # lazy: only this path needs h5py
    import h5py

    work = cfg.paths.work
    partial = work / "build_partial.npz"

    rows_aoa, rows_style, rows_perf, rows_geom, rows_rmse, rows_sid = (
        [], [], [], [], [], [])
    n_done = 0
    geom_cache: dict[int, tuple] = {}
    use_direct_cst = False
    cst_coeffs = None

    # Check if CST coefficients are available directly in the dataset
    files = find_h5_files(cfg.paths.raw_data)
    if files:
        with h5py.File(files[0], "r") as f:
            cst_coeffs = _get_cst_coefficients(f, cfg)
            if cst_coeffs is not None:
                use_direct_cst = True
                log.info("Using pre-computed CST coefficients from dataset (%d shapes)", 
                        len(cst_coeffs))

    if resume and partial.exists():
        try:
            d = np.load(partial, allow_pickle=False)
            rows_aoa = list(d["aoa"]); rows_style = list(d["style"])
            rows_perf = list(d["perf"]); rows_geom = list(d["geom"])
            rows_rmse = list(d["rmse"]); rows_sid = list(d["sid"])
            n_done = int(d["n_done"])
            log.info("Resuming build from checkpoint at %d records", n_done)
        except Exception as e:
            log.warning("Could not load checkpoint: %s. Starting fresh.", e)
            partial.unlink(missing_ok=True)

    g = cfg.cst
    no_flow_warned = False

    record_iter = iter_records(cfg, max_shapes=max_shapes)
    for i, rec in enumerate(tqdm(record_iter, desc="Extracting features")):
        if i < n_done:
            continue

        # Geometry (cache per shape; identical across AoA).
        if rec.shape_id in geom_cache:
            geom_vec, rmse = geom_cache[rec.shape_id]
        else:
            if use_direct_cst and cst_coeffs is not None:
                # Use pre-computed CST coefficients directly from the dataset
                cst_vec = cst_coeffs[rec.shape_id]
                if cst_vec.ndim == 2:  # (2, n_coeffs) format - upper/lower surfaces
                    # Flatten upper and lower CST coefficients
                    geom_vec = np.concatenate([cst_vec[0], cst_vec[1]])
                else:  # Already flat vector
                    geom_vec = cst_vec
                
                # No fitting error since coefficients come directly from the dataset
                rmse = 0.0
            else:
                # Fall back to CST fitting from landmarks
                geom_vec, _yte, rmse = fit_airfoil(
                    rec.landmarks, n_order=g.n_order,
                    n_surface_pts=g.n_surface_pts, n1=g.n1, n2=g.n2)
            
            geom_cache[rec.shape_id] = (geom_vec, rmse)

        # Style.
        cp = _surface_cp(rec, cfg)
        if cp is None:
            if not no_flow_warned:
                log.warning("No flow field for some records; style set to NaN "
                            "(forward-direction training still works).")
                no_flow_warned = True
            style = np.full(4, np.nan)
        else:
            style = compute_style_features(cp, rec.landmarks)

        rows_aoa.append(rec.aoa)
        rows_style.append(style)
        rows_perf.append([rec.cl, rec.cd, rec.cm])
        rows_geom.append(geom_vec)
        rows_rmse.append(rmse)
        rows_sid.append(rec.shape_id)

        if (i + 1) % cfg.checkpoint_every == 0:
            _checkpoint(partial, rows_aoa, rows_style, rows_perf,
                        rows_geom, rows_rmse, rows_sid, i + 1)

    out = {
        "aoa": np.asarray(rows_aoa, dtype=np.float32).reshape(-1, 1),
        "style": np.asarray(rows_style, dtype=np.float32),
        "perf": np.asarray(rows_perf, dtype=np.float32),
        "geom": np.asarray(rows_geom, dtype=np.float32),
        "rmse": np.asarray(rows_rmse, dtype=np.float32),
        "sid": np.asarray(rows_sid, dtype=np.int32),
    }
    for name, arr in out.items():
        save_npy(work / f"{name}.npy", arr)
    log.info("Built %d records -> %s", len(rows_aoa), work)
    if partial.exists():
        partial.unlink()
    return out


def _checkpoint(path, aoa, style, perf, geom, rmse, sid, n_done):
    """Save checkpoint using explicit file handle to avoid NumPy's .npz extension issue on Windows."""
    import os
    from pathlib import Path

    tmp = Path(str(path) + ".tmp")

    # Open the file explicitly so NumPy doesn't append ".npz"
    with open(tmp, "wb") as f:
        np.savez(
            f,
            aoa=np.asarray(aoa, np.float32),
            style=np.asarray(style, np.float32),
            perf=np.asarray(perf, np.float32),
            geom=np.asarray(geom, np.float32),
            rmse=np.asarray(rmse, np.float32),
            sid=np.asarray(sid, np.int32),
            n_done=np.int64(n_done),
        )

    os.replace(tmp, path)


def assemble_xy(cfg: Config, arrays: dict | None = None):
    """Build the (X, Y) matrices for the configured direction.

    inverse: X = [AoA | style(4)]            Y = [geom | CL CD CM]
    forward: X = [geom | AoA]                Y = [CL CD CM]
    """
    work = cfg.paths.work
    if arrays is None:
        arrays = {k: np.load(work / f"{k}.npy")
                  for k in ("aoa", "style", "perf", "geom", "rmse", "sid")}

    aoa, style, perf = arrays["aoa"], arrays["style"], arrays["perf"]
    geom, rmse = arrays["geom"], arrays["rmse"]

    if cfg.rbf.direction == "forward":
        X = np.hstack([geom, aoa]).astype(np.float32)
        Y = perf.astype(np.float32)
    else:  # inverse (Clark's map)
        X = np.hstack([aoa, style]).astype(np.float32)
        Y = np.hstack([geom, perf]).astype(np.float32)

    save_npy(work / "X_raw.npy", X)
    save_npy(work / "Y_raw.npy", Y)
    log.info("Assembled X%s and Y%s for direction=%s",
             X.shape, Y.shape, cfg.rbf.direction)
    return X, Y, rmse