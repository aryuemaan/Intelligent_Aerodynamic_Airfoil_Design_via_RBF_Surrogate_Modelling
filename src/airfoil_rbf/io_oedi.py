"""Discovery and streaming access to the NREL windAI_bench 9k airfoil data.

The dataset is distributed as HDF5 under
    s3://nrel-pds-windai/aerodynamic_shapes/2D/9k_airfoils/
totalling ~52.7 GB. Each shape has a `landmarks` geometry array and, per
AoA, scalar CL/CD/CM plus per-node conservative flow variables.

Exact internal naming varies by release, so this module:
  * auto-discovers .h5 files under the raw-data root,
  * walks each file's group tree to locate per-(shape, AoA) records,
  * resolves dataset names against the candidate lists in config.HDF5Keys,
  * yields lightweight ShapeRecord objects WITHOUT holding the whole file.

If your local layout differs, run `python -m airfoil_rbf inspect` to print
the real tree, then adjust config.yaml `keys:` accordingly. The synthetic
generator (`make-synthetic`) writes the canonical layout this loader treats
as the reference.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import h5py
import numpy as np

from .config import Config
from .utils import resolve_key, log

_AOA_RE = re.compile(r"(-?\d+(?:\.\d+)?)")


@dataclass
class ShapeRecord:
    shape_id: int
    aoa: float
    landmarks: np.ndarray            # (Ns, 2)
    node_xy: Optional[np.ndarray]    # (Nn, 2) or None if not available
    density: Optional[np.ndarray]
    momentum_x: Optional[np.ndarray]
    momentum_y: Optional[np.ndarray]
    energy: Optional[np.ndarray]
    cl: float
    cd: float
    cm: float

    @property
    def has_flow(self) -> bool:
        return (self.node_xy is not None and self.density is not None
                and self.momentum_x is not None and self.momentum_y is not None)


def find_h5_files(root: Path) -> List[Path]:
    files = sorted(p for p in Path(root).rglob("*.h5"))
    files += sorted(p for p in Path(root).rglob("*.hdf5"))
    return files


def _parse_aoa(name: str) -> Optional[float]:
    m = _AOA_RE.search(name)
    return float(m.group(1)) if m else None


def _read(group, candidates, keys_obj):
    key = resolve_key(group, getattr(keys_obj, candidates))
    if key is None:
        return None
    return np.asarray(group[key][()])


def _scalar(group, candidates, keys_obj, default=np.nan):
    arr = _read(group, candidates, keys_obj)
    if arr is None:
        return float(default)
    return float(np.asarray(arr).ravel()[0])


def _looks_like_record(group, keys) -> bool:
    """A leaf group is a (shape, AoA) record if it exposes CL or flow vars."""
    return (resolve_key(group, keys.cl) is not None
            or resolve_key(group, keys.density) is not None)


def _node_xy(group, keys, landmarks):
    """Best-effort node coordinate recovery."""
    xk = resolve_key(group, keys.node_x)
    yk = resolve_key(group, keys.node_y)
    if xk is not None and yk is not None:
        return np.column_stack([np.asarray(group[xk][()]).ravel(),
                                np.asarray(group[yk][()]).ravel()])
    # Some releases pack node coords as a (Nn, >=2) "coordinates"/"mesh" array.
    for cand in ("coordinates", "mesh", "nodes", "xy"):
        if cand in group:
            arr = np.asarray(group[cand][()])
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr[:, :2]
    return None


def iter_records(cfg: Config, max_shapes: Optional[int] = None) -> Iterator[ShapeRecord]:
    """Yield ShapeRecord objects across all discovered files.

    Handles four common layouts:
      A) one file, groups /<shape>/<aoa>/...     (canonical / synthetic)
      B) per-shape files <shape>.h5 with /<aoa>/...
      C) flat file with a 'landmarks' stack + parallel CL/CD arrays
         indexed by shape (loader falls back to index alignment).
      D) NREL 9k layout: /shape/landmarks + /alpha+XX/{C_l,C_d,C_m,flow_field/####}
    """
    keys = cfg.keys
    files = find_h5_files(cfg.paths.raw_data)
    if not files:
        raise FileNotFoundError(
            f"No .h5/.hdf5 files under {cfg.paths.raw_data}. "
            f"Download the dataset first (see download_data.sh) or run "
            f"`python -m airfoil_rbf make-synthetic`."
        )
    log.info("Discovered %d HDF5 file(s) under %s", len(files), cfg.paths.raw_data)

    emitted = 0
    sid_counter = 0

    for fpath in files:
        with h5py.File(fpath, "r") as f:
            # ----- NREL 9k flat-layout support -----
            if (
                "shape" in f
                and "alpha+04" in f
                and "alpha+12" in f
            ):
                shape = f["shape"]
                landmarks = np.asarray(shape["landmarks"])

                for aoa_name in ("alpha+04", "alpha+12"):
                    grp = f[aoa_name]
                    aoa = 4.0 if "04" in aoa_name else 12.0

                    cl = np.asarray(grp["C_l"])
                    cd = np.asarray(grp["C_d"])
                    cm = np.asarray(grp["C_m"])

                    flow = grp["flow_field"]

                    for i in range(len(cl)):
                        if max_shapes and emitted >= max_shapes * 2:  # 2 AoAs per shape
                            return

                        sid = f"{i:04d}"
                        ff = flow[sid]

                        node_xy = np.column_stack(
                            [
                                np.asarray(ff["x"]),
                                np.asarray(ff["y"]),
                            ]
                        )

                        yield ShapeRecord(
                            shape_id=i,
                            aoa=aoa,
                            landmarks=landmarks[i],
                            node_xy=node_xy,
                            density=np.asarray(ff["rho"]),
                            momentum_x=np.asarray(ff["rho_u"]),
                            momentum_y=np.asarray(ff["rho_v"]),
                            energy=np.asarray(ff["e"]),
                            cl=float(cl[i]),
                            cd=float(cd[i]),
                            cm=float(cm[i]),
                        )
                        emitted += 1

                return  # NREL 9k layout handled, exit

            # Original layouts A/B/C
            shape_groups = _enumerate_shape_groups(f, keys)
            for sid, shape_grp, landmark_arr in shape_groups:
                lm = landmark_arr
                if lm is None:
                    lm = _read(shape_grp, "landmarks", keys)
                if lm is None:
                    continue
                lm = np.asarray(lm, dtype=float)
                if lm.ndim != 2 or lm.shape[1] < 2:
                    continue

                for aoa, rec_grp in _enumerate_aoa_groups(shape_grp, keys, cfg.flow.aoas):
                    node_xy = _node_xy(rec_grp, keys, lm)
                    rec = ShapeRecord(
                        shape_id=sid if sid is not None else sid_counter,
                        aoa=aoa,
                        landmarks=lm,
                        node_xy=node_xy,
                        density=_read(rec_grp, "density", keys),
                        momentum_x=_read(rec_grp, "momentum_x", keys),
                        momentum_y=_read(rec_grp, "momentum_y", keys),
                        energy=_read(rec_grp, "energy", keys),
                        cl=_scalar(rec_grp, "cl", keys),
                        cd=_scalar(rec_grp, "cd", keys),
                        cm=_scalar(rec_grp, "cm", keys),
                    )
                    yield rec
                    emitted += 1
                sid_counter += 1
                if max_shapes and sid_counter >= max_shapes:
                    return
    if emitted == 0:
        raise RuntimeError(
            "Found HDF5 files but could not parse any (shape, AoA) records. "
            "Run `python -m airfoil_rbf inspect` and update config.yaml keys."
        )


def _enumerate_shape_groups(f, keys):
    """Yield (shape_id, group, landmarks_or_None) for each shape in a file."""
    top_keys = list(f.keys())

    # Layout C: flat file with a landmark stack at the root.
    root_lm = resolve_key(f, keys.landmarks)
    if root_lm is not None and isinstance(f[root_lm], h5py.Dataset):
        stack = np.asarray(f[root_lm][()])
        if stack.ndim == 3:                      # (Nshapes, Ns, 2)
            for i in range(stack.shape[0]):
                yield i, f, stack[i]
            return

    # Layouts A/B: groups, one per shape (or the file itself is one shape).
    group_children = [k for k in top_keys if isinstance(f[k], h5py.Group)]
    if not group_children:
        yield None, f, None                      # file == single shape
        return

    # If children look like AoA groups, the file itself is one shape.
    if all(_parse_aoa(k) is not None for k in group_children) and \
            any(_looks_like_record(f[k], keys) for k in group_children):
        yield None, f, None
        return

    for k in group_children:
        sid = _shape_id_from_name(k)
        yield sid, f[k], None


def _enumerate_aoa_groups(shape_grp, keys, configured_aoas):
    """Yield (aoa, record_group) for each AoA under a shape group.

    Falls back to treating the shape group itself as a single record if no
    AoA subgroups are present.
    """
    sub = [k for k in shape_grp.keys() if isinstance(shape_grp[k], h5py.Group)]
    aoa_subs = [(k, _parse_aoa(k)) for k in sub]
    aoa_subs = [(k, a) for k, a in aoa_subs if a is not None]

    if aoa_subs:
        for name, aoa in aoa_subs:
            yield aoa, shape_grp[name]
        return

    # No AoA subgroups: the record group may carry an AoA attribute/dataset.
    aoa = _scalar(shape_grp, "cl", keys, default=np.nan)  # dummy to test access
    aoa_val = None
    for cand in ("AoA", "aoa", "alpha", "angle_of_attack"):
        if cand in shape_grp:
            aoa_val = float(np.asarray(shape_grp[cand][()]).ravel()[0])
            break
    if aoa_val is None and "AoA" in shape_grp.attrs:
        aoa_val = float(shape_grp.attrs["AoA"])
    yield (aoa_val if aoa_val is not None else configured_aoas[0]), shape_grp


def _shape_id_from_name(name: str) -> Optional[int]:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else None