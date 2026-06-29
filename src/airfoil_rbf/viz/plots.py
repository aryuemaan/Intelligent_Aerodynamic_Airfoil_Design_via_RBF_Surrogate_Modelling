"""Matplotlib helpers. Uses the Agg backend so plots save to disk without a
display (important on headless Windows/servers and inside Git Bash)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..geometry import reconstruct_airfoil


def plot_airfoil(xy, path, title="Airfoil"):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(xy[:, 0], xy[:, 1], "-", lw=1.5, color="#0066b3")
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("x/c"); ax.set_ylabel("y/c"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_cst_fit(landmarks, geom_vec, path, title="CST fit"):
    psi, y_u, y_l = reconstruct_airfoil(geom_vec)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(landmarks[:, 0], landmarks[:, 1], ".", ms=2, color="0.6",
            label="landmarks")
    ax.plot(psi, y_u, "-", color="#cc3333", lw=1.4, label="CST upper")
    ax.plot(psi, y_l, "-", color="#0066b3", lw=1.4, label="CST lower")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_xlabel("x/c"); ax.set_ylabel("y/c"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_parity(y_true, y_pred, labels, path):
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, t, p, lab in zip(axes, y_true.T, y_pred.T, labels):
        ax.scatter(t, p, s=6, alpha=0.3, color="#0066b3")
        lims = [min(t.min(), p.min()), max(t.max(), p.max())]
        ax.plot(lims, lims, "r--", lw=1.3)
        ax.set_xlabel(f"True {lab}"); ax.set_ylabel(f"Pred {lab}")
        ax.set_title(lab); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_airfoil_family(families, path, title="Airfoil family"):
    """families: list of dicts with keys psi, y_upper, y_lower, label."""
    fig, ax = plt.subplots(figsize=(10, 4))
    cmap = plt.get_cmap("viridis")
    for i, fam in enumerate(families):
        c = cmap(i / max(len(families) - 1, 1))
        ax.plot(fam["psi"], fam["y_upper"], color=c, lw=1.3,
                label=fam.get("label", ""))
        ax.plot(fam["psi"], fam["y_lower"], color=c, lw=1.3)
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_xlabel("x/c"); ax.set_ylabel("y/c"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
