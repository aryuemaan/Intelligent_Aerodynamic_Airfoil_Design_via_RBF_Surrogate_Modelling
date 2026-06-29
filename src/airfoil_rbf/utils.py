"""Small shared helpers: logging, timing, atomic numpy save, key resolution."""
from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# HDF5 file locking frequently breaks on Windows / network drives.
# Disable it before h5py is imported anywhere.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def get_logger(name: str = "airfoil_rbf") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                                datefmt="%H:%M:%S")
        h.setFormatter(fmt)
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


log = get_logger()


@contextmanager
def timer(label: str):
    t0 = time.time()
    yield
    log.info("%s took %.2f s", label, time.time() - t0)


def save_npy(path: str | os.PathLike, arr: np.ndarray) -> None:
    """Atomic .npy save (write temp then rename) to survive interruptions.

    We write through an explicit file handle because ``np.save`` silently
    appends ``.npy`` to any path whose name does not already end in ``.npy``
    (which would desync the temp name from the rename target).
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
    os.replace(tmp, path)  # atomic on the same filesystem, incl. Windows


def resolve_key(container, candidates: Sequence[str]) -> str | None:
    """Return the first candidate key present in an h5py group / dict."""
    keys = set(container.keys())
    for c in candidates:
        if c in keys:
            return c
    # case-insensitive fallback
    lower = {k.lower(): k for k in keys}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def chunks(seq: Sequence, size: int) -> Iterable[Sequence]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]
