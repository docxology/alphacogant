#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from matplotlib.collections import LineCollection

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alphacogant.viz.plot_style import (  # noqa: E402
    CHANNEL_COLORS,
    EPISTEMIC_COLOR,
    PRAGMATIC_COLOR,
)


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = PROJECT_ROOT / "output" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(314159)
    theta = np.linspace(0, 2 * np.pi, 480)
    radius = 1.0 + 0.16 * np.sin(5 * theta) + 0.08 * np.cos(8 * theta)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    fig, ax = plt.subplots(figsize=(8.0, 10.5), dpi=240)
    fig.patch.set_facecolor("#07111f")
    ax.set_facecolor("#07111f")

    grid_x, grid_y = np.meshgrid(np.linspace(-1.35, 1.35, 520), np.linspace(-1.65, 1.65, 680))
    field = (
        np.sin(5.2 * grid_x + 2.0 * grid_y)
        + np.cos(4.6 * grid_y - 1.2 * grid_x)
        + 0.7 * np.sin(8.0 * np.hypot(grid_x, grid_y))
    )
    ax.imshow(
        field,
        extent=(-1.35, 1.35, -1.65, 1.65),
        origin="lower",
        cmap="magma",
        alpha=0.30,
        interpolation="bilinear",
    )

    points = np.column_stack([x, y])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    colours = np.linspace(0, 1, len(segments))
    ax.add_collection(
        LineCollection(
            segments,
            cmap="viridis",
            array=colours,
            linewidths=4.2,
            alpha=0.94,
        )
    )

    channel_angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 6)[:-1]
    channel_names = ["I", "S", "U", "Theta", "Z"]
    channel_points = []
    for index, (name, angle) in enumerate(zip(channel_names, channel_angles, strict=True)):
        point = np.array([1.02 * np.cos(angle), 1.02 * np.sin(angle)])
        channel_points.append(point)
        ax.scatter(
            point[0],
            point[1],
            s=360,
            color=CHANNEL_COLORS[name],
            edgecolor="white",
            linewidth=1.4,
            zorder=4,
        )
        label = r"$\Theta$" if name == "Theta" else name
        ax.text(
            point[0],
            point[1],
            label,
            color="white",
            ha="center",
            va="center",
            fontsize=14 if name != "Theta" else 13,
            fontweight="bold",
            zorder=5,
        )
        for target in channel_points[:index]:
            curve = np.linspace(0, 1, 100)
            midpoint = 0.18 * rng.normal(size=2)
            line = (
                ((1 - curve) ** 2)[:, None] * point
                + (2 * (1 - curve) * curve)[:, None] * midpoint
                + (curve**2)[:, None] * target
            )
            ax.plot(line[:, 0], line[:, 1], color="#dbeafe", alpha=0.10, linewidth=0.9)

    for scale, alpha, color in [
        (0.74, 0.38, EPISTEMIC_COLOR),
        (0.52, 0.32, PRAGMATIC_COLOR),
        (0.31, 0.24, "#38bdf8"),
    ]:
        ax.plot(scale * x, scale * y, color=color, alpha=alpha, linewidth=1.8)

    ax.scatter([0], [0], s=700, color="#f8fafc", edgecolor="#38bdf8", linewidth=2.0, zorder=6)
    ax.text(
        0,
        0.015,
        "G",
        ha="center",
        va="center",
        fontsize=24,
        color="#0f172a",
        fontweight="bold",
        zorder=7,
    )
    ax.set_xlim(-1.32, 1.32)
    ax.set_ylim(-1.62, 1.58)
    ax.axis("off")
    fig.tight_layout(pad=0)

    output_path = output_dir / "cover_art.png"
    fig.savefig(
        output_path,
        dpi=240,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)
    print(str(output_path.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
