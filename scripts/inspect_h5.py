"""Standalone HDF5 inspector (works without the package on PYTHONPATH).

    python scripts/inspect_h5.py path/to/file.h5
"""
import sys
from pathlib import Path

import h5py


def walk(node, prefix="", depth=0, max_depth=5, max_items=15):
    if depth > max_depth:
        return
    for i, k in enumerate(node.keys()):
        if i >= max_items:
            print(f"{prefix}... (+{len(node.keys()) - max_items} more)")
            break
        item = node[k]
        if isinstance(item, h5py.Group):
            print(f"{prefix}{k}/  (group)")
            walk(item, prefix + "  ", depth + 1, max_depth, max_items)
        else:
            print(f"{prefix}{k}  shape={item.shape} dtype={item.dtype}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python scripts/inspect_h5.py <file.h5>")
        raise SystemExit(1)
    path = Path(sys.argv[1])
    with h5py.File(path, "r") as f:
        print(path)
        if f.attrs:
            print("attrs:", dict(f.attrs))
        walk(f)
