"""Generate lightweight comparison figures for the sdf_to_density converters.

Produces two PNGs under ``website/public/images/guides/``:

- ``sdf-to-density-profiles.png`` — 1D profile plot comparing the three
  conversion profiles (heaviside, linear, sigmoid) at bandwidth=1, with
  dashed overlays at bandwidth=2 to show the bandwidth knob.
- ``sdf-to-density-sphere.png`` — three side-by-side renders of an
  iso-surface extracted from a 24³ sphere SDF after each conversion method,
  with smoothing/repair/remesh/decimate disabled so the panels show the
  raw difference attributable to the converter.

Usage:
    uv run python scripts/generate_converter_comparison.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from xeltofab import heaviside, linear_ramp, sdf_to_density, sigmoid
from xeltofab.extract import extract
from xeltofab.sdf_eval import uniform_grid_evaluate
from xeltofab.state import PipelineParams, PipelineState

OUTPUT_DIR = Path("website/public/images/guides")
DPI = 150
BG_COLOR = "white"

HEAVISIDE_COLOR = "#e45756"
LINEAR_COLOR = "#4c78a8"
SIGMOID_COLOR = "#59a14f"
MESH_COLOR = "#88BDE6"


def _sphere_sdf(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points, axis=1) - 1.0


def gen_profiles() -> None:
    sdf = np.linspace(-3.0, 3.0, 601)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=DPI)

    ax.axhline(0.5, color="gray", lw=0.6, ls=":", zorder=0)
    ax.axvline(0.0, color="gray", lw=0.6, ls=":", zorder=0)
    ax.axvspan(-1.0, 1.0, color=LINEAR_COLOR, alpha=0.07, zorder=0,
               label="linear band (bw=1)")

    ax.plot(sdf, heaviside(sdf), color=HEAVISIDE_COLOR, lw=2.2,
            label="heaviside", zorder=3)
    ax.plot(sdf, linear_ramp(sdf, bandwidth=1.0), color=LINEAR_COLOR, lw=2.2,
            label="linear  (bw=1)", zorder=3)
    ax.plot(sdf, sigmoid(sdf, bandwidth=1.0), color=SIGMOID_COLOR, lw=2.2,
            label="sigmoid (bw=1)", zorder=3)

    ax.plot(sdf, linear_ramp(sdf, bandwidth=2.0), color=LINEAR_COLOR, lw=1.4,
            ls="--", label="linear  (bw=2)", zorder=2)
    ax.plot(sdf, sigmoid(sdf, bandwidth=2.0), color=SIGMOID_COLOR, lw=1.4,
            ls="--", label="sigmoid (bw=2)", zorder=2)

    ax.annotate("iso-level invariant:\nsdf = level → density = 0.5",
                xy=(0.0, 0.5), xytext=(-2.85, 0.12),
                fontsize=8.5, color="#444", ha="left",
                arrowprops={"arrowstyle": "->", "color": "#888", "lw": 0.7,
                            "connectionstyle": "arc3,rad=0.25"})

    ax.set_xlabel("sdf  (signed distance; level = 0)")
    ax.set_ylabel("density")
    ax.set_title("sdf_to_density — conversion profiles")
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-0.05, 1.08)
    ax.legend(frameon=False, loc="upper right", fontsize=8.5, ncol=1)
    ax.grid(True, alpha=0.15)

    fig.patch.set_facecolor(BG_COLOR)
    fig.tight_layout()
    out = OUTPUT_DIR / "sdf-to-density-profiles.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def _render_mesh(vertices: np.ndarray, faces: np.ndarray,
                 window_size: tuple[int, int] = (600, 600)) -> np.ndarray:
    """Off-screen isometric render of a mesh, matched framing for comparison."""
    faces_pv = np.column_stack([np.full(len(faces), 3), faces]).ravel()
    pv_mesh = pv.PolyData(vertices.astype(np.float64), faces_pv)

    pl = pv.Plotter(off_screen=True, window_size=list(window_size))
    pl.add_mesh(pv_mesh, color=MESH_COLOR, show_edges=True,
                edge_color="#1f3a4b", line_width=0.4)
    pl.set_background(BG_COLOR)
    pl.camera_position = "iso"
    pl.camera.zoom(1.3)
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


def _extract_only(density: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run extraction in isolation — no smoothing, repair, remesh, or decimation."""
    params = PipelineParams(
        field_type="density",
        direct_extraction=True,
        extraction_level=0.5,
        extraction_method="mc",
        repair=False,
        remesh=False,
        decimate=False,
    )
    state = extract(PipelineState(field=density, params=params))
    assert state.vertices is not None and state.faces is not None
    return state.vertices, state.faces


def gen_sphere_comparison() -> None:
    bounds = (-1.3, -1.3, -1.3, 1.3, 1.3, 1.3)
    resolution = 24
    sdf_grid, *_ = uniform_grid_evaluate(_sphere_sdf, bounds, resolution=resolution)

    voxel = (bounds[3] - bounds[0]) / (resolution - 1)
    bw = 2.0 * voxel

    panels: list[tuple[str, str, np.ndarray]] = []
    for method, subtitle in (
        ("heaviside", "binary density"),
        ("linear", f"bandwidth = 2·voxel ({bw:.3f})"),
        ("sigmoid", f"bandwidth = 2·voxel ({bw:.3f})"),
    ):
        density = sdf_to_density(sdf_grid, method=method, bandwidth=bw)  # type: ignore[arg-type]
        verts, faces = _extract_only(density)
        img = _render_mesh(verts, faces)
        panels.append((method, f"{subtitle}  —  {len(faces)} faces", img))

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.3), dpi=DPI)
    for ax, (method, subtitle, img) in zip(axes, panels, strict=True):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(method, fontsize=11, fontweight="bold", pad=6)
        ax.text(0.5, -0.04, subtitle, transform=ax.transAxes,
                ha="center", va="top", fontsize=8.5, color="#555")

    fig.suptitle(
        "Sphere SDF (radius 1, resolution 24³) → density → MC iso=0.5",
        fontsize=10, y=1.02, color="#444",
    )
    fig.patch.set_facecolor(BG_COLOR)
    fig.tight_layout()
    out = OUTPUT_DIR / "sdf-to-density-sphere.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def gen_field_slice() -> None:
    """2D density-field slices — shows where linear and sigmoid actually diverge.

    Matches the sphere example from the mesh comparison but plots the density
    field on a fine 2D grid instead of the extracted mesh. This separates the
    profile shape (visible) from MC vertex placement (nearly identical).
    """
    extent = 1.5
    n = 128
    g = np.linspace(-extent, extent, n)
    xx, yy = np.meshgrid(g, g, indexing="ij")
    sdf2d = np.sqrt(xx**2 + yy**2) - 1.0  # unit circle SDF in 2D

    bw = 0.3

    panels: list[tuple[str, str, np.ndarray]] = []
    for method, subtitle in (
        ("heaviside", "binary — density ∈ {0, 0.5, 1}"),
        ("linear", f"compact support, bw = {bw}"),
        ("sigmoid", f"infinite tails, bw = {bw}"),
    ):
        density = sdf_to_density(sdf2d, method=method, bandwidth=bw)  # type: ignore[arg-type]
        panels.append((method, subtitle, density))

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.3), dpi=DPI)
    for ax, (method, subtitle, density) in zip(axes, panels, strict=True):
        im = ax.imshow(
            density.T, origin="lower", extent=(-extent, extent, -extent, extent),
            cmap="magma", vmin=0.0, vmax=1.0,
        )
        ax.contour(xx, yy, density, levels=[0.5], colors="#ffffff",
                   linewidths=1.2, linestyles="--")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(method, fontsize=11, fontweight="bold", pad=6)
        ax.text(0.5, -0.04, subtitle, transform=ax.transAxes,
                ha="center", va="top", fontsize=8.5, color="#555")

    cbar = fig.colorbar(im, ax=axes, shrink=0.75, aspect=22, pad=0.02)
    cbar.set_label("density", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        "2D density field for a unit-circle SDF — white dashed line is the MC iso=0.5 contour",
        fontsize=10, y=0.96, color="#444",
    )
    fig.patch.set_facecolor(BG_COLOR)
    fig.subplots_adjust(top=0.88, bottom=0.08)
    out = OUTPUT_DIR / "sdf-to-density-field-slice.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gen_profiles()
    gen_sphere_comparison()
    gen_field_slice()


if __name__ == "__main__":
    main()
