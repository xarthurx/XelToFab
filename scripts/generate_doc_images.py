"""Generate documentation images for website guide pages.

Usage:
    uv run python scripts/generate_doc_images.py [--only NAME]

Valid --only values: pipeline_diagram, pipeline_stages, field_types,
    parameter_sensitivity, quality_metrics
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Shared constants
OUTPUT_DIR = Path("website/public/images/guides")
DPI = 150
BG_COLOR = "white"

# Pipeline stage colors (reused across images 1 and 2)
STAGE_COLORS = {
    "Preprocess": "#4A90D9",
    "Extract": "#50B86C",
    "Smooth": "#F5A623",
    "Repair": "#D0021B",
    "Remesh": "#9B59B6",
    "Decimate": "#1ABC9C",
}


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pv_screenshot(vertices, faces, color="steelblue"):
    """Render a mesh off-screen and return the image as a numpy array."""
    import numpy as np
    import pyvista as pv

    faces_pv = np.column_stack([np.full(len(faces), 3), faces]).ravel()
    pv_mesh = pv.PolyData(vertices.astype(np.float64), faces_pv)

    pl = pv.Plotter(off_screen=True, window_size=[512, 512])
    pl.add_mesh(pv_mesh, color=color, show_edges=True, edge_color="black", line_width=0.3)
    pl.camera_position = "iso"
    pl.set_background(BG_COLOR)
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


# ---------------------------------------------------------------------------
# Image generators (one per function)
# ---------------------------------------------------------------------------


def gen_pipeline_diagram() -> None:
    """Image 1: Horizontal pipeline flow diagram with 6 colored stage boxes."""
    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.set_xlim(-0.5, 12.0)
    ax.set_ylim(-1.2, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(BG_COLOR)

    # Terminals (rounded, gray)
    terminals = [("Field", 0.0), ("Mesh", 11.5)]
    for label, x in terminals:
        ax.add_patch(matplotlib.patches.FancyBboxPatch(
            (x - 0.45, -0.3), 0.9, 0.6,
            boxstyle="round,pad=0.1",
            facecolor="#E0E0E0", edgecolor="#666666", linewidth=1.5,
        ))
        ax.text(x, 0.0, label, ha="center", va="center", fontsize=10, fontweight="bold")

    # Stage boxes (colored)
    stages = list(STAGE_COLORS.keys())
    positions = [1.7, 3.4, 5.1, 6.8, 8.5, 10.0]
    param_annotations = {
        "Preprocess": "threshold\nsmooth_sigma",
        "Extract": "extraction_level",
        "Smooth": "taubin_iterations\nsmoothing_method",
        "Repair": "repair",
        "Remesh": "target_edge_length\nremesh_iterations",
        "Decimate": "decimate_ratio\ndecimate_aggressiveness",
    }

    for stage, x in zip(stages, positions, strict=True):
        color = STAGE_COLORS[stage]
        ax.add_patch(matplotlib.patches.FancyBboxPatch(
            (x - 0.55, -0.3), 1.1, 0.6,
            boxstyle="round,pad=0.08",
            facecolor=color, edgecolor="black", linewidth=1.2, alpha=0.85,
        ))
        ax.text(x, 0.0, stage, ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="white")

        # Parameter annotation below
        if stage in param_annotations:
            ax.text(x, -0.65, param_annotations[stage], ha="center", va="top",
                    fontsize=5.5, color="#555555", fontstyle="italic")

    # Arrows between all boxes: Field -> Preprocess -> ... -> Decimate -> Mesh
    all_x = [0.0] + positions + [11.5]
    for i in range(len(all_x) - 1):
        x_start = all_x[i] + 0.55
        x_end = all_x[i + 1] - 0.55
        ax.annotate(
            "", xy=(x_end, 0.0), xytext=(x_start, 0.0),
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.5),
        )

    fig.savefig(OUTPUT_DIR / "pipeline-flow.png", dpi=DPI, bbox_inches="tight",
                facecolor=BG_COLOR)
    plt.close(fig)


def gen_pipeline_stages() -> None:
    """Image 2: Stage-by-stage visual progression."""
    pass  # implemented in Task 3


def gen_field_types() -> None:
    """Image 3: Density vs SDF 2x2 comparison."""
    pass  # implemented in Task 4


def gen_parameter_sensitivity() -> None:
    """Image 4: 3x3 parameter sensitivity grid."""
    pass  # implemented in Task 5


def gen_quality_metrics() -> None:
    """Image 5: Quality heatmap + histogram composite."""
    pass  # implemented in Task 6


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

GENERATORS = {
    "pipeline_diagram": gen_pipeline_diagram,
    "pipeline_stages": gen_pipeline_stages,
    "field_types": gen_field_types,
    "parameter_sensitivity": gen_parameter_sensitivity,
    "quality_metrics": gen_quality_metrics,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate documentation images for guide pages.")
    parser.add_argument(
        "--only",
        choices=list(GENERATORS.keys()),
        help="Generate only the specified image",
    )
    args = parser.parse_args()

    _ensure_output_dir()

    targets = [args.only] if args.only else list(GENERATORS.keys())
    for name in targets:
        print(f"Generating {name}...")
        GENERATORS[name]()
        print(f"  Done.")


if __name__ == "__main__":
    main()
