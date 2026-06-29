"""Central configuration: loads config.yaml, fills defaults, validates.

Everything tunable lives here so the pipeline scripts stay clean.
Paths are resolved relative to the project root (the directory that
contains config.yaml), which keeps things working identically under
Windows / Git Bash and Linux.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyYAML is required. Install with: pip install pyyaml"
    ) from exc


def _project_root() -> Path:
    """Find the project root by walking up until config.yaml is found."""
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "config.yaml").exists():
            return parent
    # Fall back to two levels up from this file (src/airfoil_rbf/config.py)
    return here.parents[2]


@dataclass
class Paths:
    root: Path
    raw_data: Path          # where the downloaded .h5 files live
    work: Path              # intermediate .npy artefacts
    models: Path            # saved surrogate + scalers
    figures: Path           # output plots

    def make_dirs(self) -> None:
        for p in (self.work, self.models, self.figures):
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class CSTConfig:
    n_order: int = 10           # Bernstein order -> n_order+1 coeffs / surface
    n_surface_pts: int = 101    # resample resolution per surface
    n1: float = 0.5             # class-function LE exponent (round LE)
    n2: float = 1.0             # class-function TE exponent (sharp TE)
    rmse_thresh: float = 1.0e-3 # reject fits worse than this (chord units)


@dataclass
class FlowConfig:
    mach_inf: float = 0.1
    reynolds: float = 9.0e6
    gamma: float = 1.4
    a_inf: float = 340.15       # speed of sound used in dataset normalisation
    rho_inf: float = 1.225
    aoas: tuple = (4.0, 12.0)
    # 'incompressible' (Cp = 1-(V/Vinf)^2, robust at M=0.1) or
    # 'compressible'   (Cp from pressure via conservative variables)
    cp_method: str = "incompressible"


@dataclass
class FilterConfig:
    cd_sigma: float = 3.0
    cl_sigma: float = 4.0
    require_positive_cl: bool = False  # off by default: wind airfoils can be near 0
    style_abs_max: float = 10.0


@dataclass
class RBFConfig:
    kernel: str = "multiquadric"
    epsilons: tuple = (0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0)
    smoothing: float = 0.0
    # 'inverse'  : [AoA, style] -> [geometry, CL, CD, CM]   (Clark's map)
    # 'forward'  : [geometry, AoA] -> [CL, CD, CM]           (for optimisation)
    direction: str = "inverse"
    # cap centres for tractable O(N^3) solve; None = use all training points
    max_centres: int | None = 6000
    random_state: int = 42
    test_size: float = 0.15
    val_size: float = 0.15


@dataclass
class HDF5Keys:
    """Candidate dataset names searched inside the .h5 files.

    The NREL files store per-node conservative flow variables plus a
    'landmarks' geometry array and scalar coefficients. Exact spellings
    vary by release, so each field lists several candidates; the loader
    uses the first that exists. Run `inspect` to confirm and edit here.
    """
    landmarks: tuple = ("landmarks", "geometry", "coordinates", "xy", "airfoil")
    node_x: tuple = ("x", "X", "x_coord", "coord_x", "nodes_x")
    node_y: tuple = ("y", "Y", "y_coord", "coord_y", "nodes_y")
    density: tuple = ("density", "rho", "Density", "rho_star")
    momentum_x: tuple = ("momentum_x", "momx", "rho_u", "x_momentum", "momentumX")
    momentum_y: tuple = ("momentum_y", "momy", "rho_v", "y_momentum", "momentumY")
    energy: tuple = ("energy", "E", "total_energy", "rho_E")
    vorticity: tuple = ("vorticity", "omega", "vort")
    cl: tuple = ("CL", "cl", "C_L", "lift", "coefficient_of_lift")
    cd: tuple = ("CD", "cd", "C_D", "drag", "coefficient_of_drag")
    cm: tuple = ("CM", "cm", "C_M", "moment", "coefficient_of_moment")


@dataclass
class Config:
    paths: Paths
    cst: CSTConfig = field(default_factory=CSTConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    filt: FilterConfig = field(default_factory=FilterConfig)
    rbf: RBFConfig = field(default_factory=RBFConfig)
    keys: HDF5Keys = field(default_factory=HDF5Keys)
    checkpoint_every: int = 500
    n_jobs: int = -1

    def summary(self) -> str:
        d = asdict(self)
        # Paths -> str for readability
        d["paths"] = {k: str(v) for k, v in d["paths"].items()}
        lines = ["Configuration:"]
        for section, vals in d.items():
            if isinstance(vals, dict):
                lines.append(f"  [{section}]")
                for k, v in vals.items():
                    lines.append(f"    {k}: {v}")
            else:
                lines.append(f"  {section}: {vals}")
        return "\n".join(lines)


def _merge(base: dataclass, override: Dict[str, Any]) -> None:
    """In-place override of dataclass fields from a plain dict."""
    for k, v in override.items():
        if hasattr(base, k):
            cur = getattr(base, k)
            if isinstance(v, list):
                v = tuple(v)
            setattr(base, k, v)


def load_config(path: str | os.PathLike | None = None) -> Config:
    root = _project_root()
    cfg_path = Path(path) if path else root / "config.yaml"

    raw: Dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    p = raw.get("paths", {})
    paths = Paths(
        root=root,
        raw_data=root / p.get("raw_data", "data/airfoil_cfd_9k"),
        work=root / p.get("work", "data/work"),
        models=root / p.get("models", "models"),
        figures=root / p.get("figures", "figures"),
    )

    cfg = Config(paths=paths)
    for section, obj in (
        ("cst", cfg.cst), ("flow", cfg.flow), ("filter", cfg.filt),
        ("rbf", cfg.rbf), ("keys", cfg.keys),
    ):
        if section in raw and isinstance(raw[section], dict):
            _merge(obj, raw[section])

    for top in ("checkpoint_every", "n_jobs"):
        if top in raw:
            setattr(cfg, top, raw[top])

    cfg.paths.make_dirs()
    return cfg
