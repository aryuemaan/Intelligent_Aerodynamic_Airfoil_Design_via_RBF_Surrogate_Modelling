"""Command-line interface.

Usage (from project root, with PYTHONPATH=src):
    python -m airfoil_rbf <command> [options]

Commands:
    make-synthetic   write a small synthetic dataset for testing
    inspect          print the HDF5 group/dataset tree of the raw data
    build            stream the dataset -> compact feature/target arrays
    assemble         build X/Y matrices for the configured direction
    filter           apply quality filters
    train            cross-validate epsilon, fit, save the surrogate
    evaluate         parity plots + accuracy report
    design           inverse design: AoA + style -> geometry + CL/CD/CM
    optimize         forward surrogate + DE search -> best airfoil
    all              build -> assemble -> filter -> train -> evaluate
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .config import load_config
from .utils import log


def _print_tree(node, prefix="", depth=0, max_depth=4, max_items=12):
    import h5py
    if depth > max_depth:
        return
    keys = list(node.keys())
    for k in keys[:max_items]:
        item = node[k]
        if isinstance(item, h5py.Group):
            print(f"{prefix}{k}/  (group, {len(item.keys())} children)")
            _print_tree(item, prefix + "  ", depth + 1, max_depth, max_items)
        else:
            print(f"{prefix}{k}  {item.shape} {item.dtype}")
    if len(keys) > max_items:
        print(f"{prefix}... (+{len(keys) - max_items} more)")


def cmd_inspect(cfg, args):
    import h5py
    from .io_oedi import find_h5_files
    files = find_h5_files(cfg.paths.raw_data)
    if not files:
        log.error("No .h5 files under %s", cfg.paths.raw_data)
        return
    log.info("Found %d files. Showing tree of: %s", len(files), files[0].name)
    with h5py.File(files[0], "r") as f:
        print(f"\n{files[0]}")
        _print_tree(f)
    print("\nIf dataset names differ from config.HDF5Keys candidates, "
          "edit config.yaml `keys:` to match.")


def cmd_make_synthetic(cfg, args):
    from .synthetic import generate
    generate(cfg, n_shapes=args.n)


def cmd_build(cfg, args):
    from .data.build_dataset import build_features
    build_features(cfg, max_shapes=args.max_shapes, resume=not args.no_resume)


def cmd_assemble(cfg, args):
    from .data.build_dataset import assemble_xy
    assemble_xy(cfg)


def cmd_filter(cfg, args):
    from .data.filters import apply_filters
    work = cfg.paths.work
    X = np.load(work / "X_raw.npy")
    Y = np.load(work / "Y_raw.npy")
    rmse = np.load(work / "rmse.npy")
    apply_filters(cfg, X, Y, rmse)


def cmd_train(cfg, args):
    from .training import train
    train(cfg)


def cmd_evaluate(cfg, args):
    from .training import evaluate
    evaluate(cfg)


def cmd_design(cfg, args):
    from .models.rbf import load_rbf
    from .models.scaler import MinMaxScaler
    from .models.optimize import design_from_style
    from .viz import plot_airfoil_family
    from .geometry import reconstruct_airfoil

    if cfg.rbf.direction != "inverse":
        log.error("`design` needs a model trained with direction=inverse.")
        return
    mdir = cfg.paths.models
    model = load_rbf(mdir / "rbf_model.npz")
    xs = MinMaxScaler.load(mdir / "x_scaler.npz")
    ys = MinMaxScaler.load(mdir / "y_scaler.npz")

    res = design_from_style(args.aoa, args.style, model, xs, ys,
                            cfg.cst.n_order)
    print(f"\nInverse design @ AoA={args.aoa}, style={args.style}")
    print(f"  Predicted CL={res['CL']:.4f}  CD={res['CD']:.5f}  "
          f"CM={res['CM']:.4f}")
    if res["warning"]:
        print(f"  WARNING: {res['warning']}")
    psi, y_u, y_l = reconstruct_airfoil(res["geometry"])
    plot_airfoil_family(
        [{"psi": psi, "y_upper": y_u, "y_lower": y_l,
          "label": f"AoA={args.aoa}"}],
        cfg.paths.figures / "designed_airfoil.png",
        title="Inverse-designed airfoil")
    print(f"  Saved figure -> {cfg.paths.figures / 'designed_airfoil.png'}")


def cmd_optimize(cfg, args):
    from .models.rbf import load_rbf
    from .models.scaler import MinMaxScaler
    from .models.optimize import optimize_airfoil
    from .viz import plot_airfoil_family

    if cfg.rbf.direction != "forward":
        log.error("`optimize` needs a model trained with direction=forward. "
                  "Set rbf.direction: forward in config.yaml, then re-run "
                  "assemble/filter/train.")
        return
    mdir, work = cfg.paths.models, cfg.paths.work
    model = load_rbf(mdir / "rbf_model.npz")
    xs = MinMaxScaler.load(mdir / "x_scaler.npz")
    ys = MinMaxScaler.load(mdir / "y_scaler.npz")

    geom = np.load(work / "geom.npy")
    g = geom.shape[1]
    lo, hi = geom.min(axis=0), geom.max(axis=0)
    bounds = np.column_stack([lo, hi])

    res = optimize_airfoil(model, xs, ys, bounds, aoa_deg=args.aoa,
                           objective=args.objective, target_cl=args.target_cl,
                           n_order=cfg.cst.n_order, maxiter=args.maxiter)
    print(f"\nOptimised airfoil @ AoA={args.aoa}, objective={args.objective}")
    print(f"  CL={res['CL']:.4f}  CD={res['CD']:.5f}  CM={res['CM']:.4f}  "
          f"L/D={res['LD']:.2f}")
    plot_airfoil_family(
        [{"psi": res["psi"], "y_upper": res["y_upper"],
          "y_lower": res["y_lower"], "label": f"opt L/D={res['LD']:.1f}"}],
        cfg.paths.figures / "optimized_airfoil.png",
        title="Surrogate-optimised airfoil")
    print(f"  Saved figure -> {cfg.paths.figures / 'optimized_airfoil.png'}")


def cmd_all(cfg, args):
    cmd_build(cfg, args)
    cmd_assemble(cfg, args)
    cmd_filter(cfg, args)
    cmd_train(cfg, args)
    cmd_evaluate(cfg, args)


def build_parser():
    p = argparse.ArgumentParser(prog="airfoil_rbf",
                                description="RBF airfoil design pipeline")
    p.add_argument("--config", default=None, help="path to config.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("make-synthetic"); s.add_argument("--n", type=int, default=60)
    s.set_defaults(func=cmd_make_synthetic)

    sub.add_parser("inspect").set_defaults(func=cmd_inspect)

    s = sub.add_parser("build")
    s.add_argument("--max-shapes", type=int, default=None)
    s.add_argument("--no-resume", action="store_true")
    s.set_defaults(func=cmd_build)

    sub.add_parser("assemble").set_defaults(func=cmd_assemble)
    sub.add_parser("filter").set_defaults(func=cmd_filter)
    sub.add_parser("train").set_defaults(func=cmd_train)
    sub.add_parser("evaluate").set_defaults(func=cmd_evaluate)

    s = sub.add_parser("design")
    s.add_argument("--aoa", type=float, required=True)
    s.add_argument("--style", type=float, nargs=4, required=True,
                   metavar=("V1", "V2", "V3", "V4"))
    s.set_defaults(func=cmd_design)

    s = sub.add_parser("optimize")
    s.add_argument("--aoa", type=float, default=4.0)
    s.add_argument("--objective", choices=["ld", "cl", "target_cl"], default="ld")
    s.add_argument("--target-cl", type=float, default=None)
    s.add_argument("--maxiter", type=int, default=60)
    s.set_defaults(func=cmd_optimize)

    s = sub.add_parser("all")
    s.add_argument("--max-shapes", type=int, default=None)
    s.add_argument("--no-resume", action="store_true")
    s.set_defaults(func=cmd_all)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    log.info("Project root: %s", cfg.paths.root)
    args.func(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
