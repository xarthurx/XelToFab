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

# Pipeline stage colors — monochromatic blue palette, dark to light progression
# Gives visual flow without rainbow noise
STAGE_COLORS = {
    "Preprocess": "#1B3A5C",
    "Extract": "#265E8A",
    "Smooth": "#3182BD",
    "Repair": "#6BAED6",
    "Remesh": "#9ECAE1",
    "Decimate": "#C6DBEF",
}
# Text color per stage (white on dark, dark on light)
_STAGE_TEXT = {
    "Preprocess": "white",
    "Extract": "white",
    "Smooth": "white",
    "Repair": "#1a1a1a",
    "Remesh": "#1a1a1a",
    "Decimate": "#1a1a1a",
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
    """Image 1: Horizontal pipeline flow diagram with even spacing and blue palette."""
    stages = list(STAGE_COLORS.keys())
    n_stages = len(stages)
    # Even spacing: 8 elements (Field + 6 stages + Mesh), gap=1.5 between centers
    gap = 1.5
    all_labels = ["Field"] + stages + ["Mesh"]
    all_x = [i * gap for i in range(len(all_labels))]
    total_w = all_x[-1]

    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.set_xlim(-0.8, total_w + 0.8)
    ax.set_ylim(-1.2, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(BG_COLOR)

    box_hw = 0.55  # half-width
    box_hh = 0.3   # half-height

    param_annotations = {
        "Preprocess": "threshold\nsmooth_sigma",
        "Extract": "extraction_level",
        "Smooth": "taubin_iterations\nsmoothing_method",
        "Repair": "repair",
        "Remesh": "target_edge_length\nremesh_iterations",
        "Decimate": "decimate_ratio\ndecimate_aggressiveness",
    }

    for label, x in zip(all_labels, all_x, strict=True):
        is_terminal = label in ("Field", "Mesh")
        if is_terminal:
            fc, ec, tc = "#E8E8E8", "#999999", "#333333"
        else:
            fc = STAGE_COLORS[label]
            ec = "#333333"
            tc = _STAGE_TEXT[label]

        ax.add_patch(matplotlib.patches.FancyBboxPatch(
            (x - box_hw, -box_hh), box_hw * 2, box_hh * 2,
            boxstyle="round,pad=0.08",
            facecolor=fc, edgecolor=ec, linewidth=1.2,
        ))
        ax.text(x, 0.0, label, ha="center", va="center",
                fontsize=9 if is_terminal else 8.5, fontweight="bold", color=tc)

        # Parameter annotation below stage boxes
        if label in param_annotations:
            ax.text(x, -box_hh - 0.35, param_annotations[label],
                    ha="center", va="top", fontsize=5.5, color="#666666",
                    fontstyle="italic")

    # Arrows
    for i in range(len(all_x) - 1):
        ax.annotate(
            "", xy=(all_x[i + 1] - box_hw - 0.05, 0.0),
            xytext=(all_x[i] + box_hw + 0.05, 0.0),
            arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.3),
        )

    fig.savefig(OUTPUT_DIR / "pipeline-flow.png", dpi=DPI, bbox_inches="tight",
                facecolor=BG_COLOR)
    plt.close(fig)


def gen_pipeline_stages() -> None:
    """Image 2: 1x4 grid showing pipeline progression on corner 3D model."""
    import numpy as np
    from xeltofab.io import load_field
    from xeltofab.preprocess import preprocess
    from xeltofab.extract import extract
    from xeltofab.smooth import smooth

    # Use corner model — simpler geometry, easier to parse visually
    state = load_field("data/examples/corner_3d_20x20x20_vf50_sample0.npy")

    # Run stages individually to capture intermediates
    state_pre = preprocess(state)
    state_ext = extract(state_pre)
    state_smo = smooth(state_ext)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
    titles = ["Raw Field\n(center slice)", "After Threshold\n(binary)",
              "Marching Cubes\n(raw mesh)", "After Smoothing"]

    # Panel 1: Center slice of raw density field
    field = state.field
    mid = field.shape[0] // 2
    axes[0].imshow(field[mid], cmap="viridis", origin="lower")
    axes[0].set_title(titles[0], fontsize=9)
    axes[0].axis("off")

    # Panel 2: Center slice of binary field
    assert state_pre.binary is not None
    axes[1].imshow(state_pre.binary[mid], cmap="gray", origin="lower")
    axes[1].set_title(titles[1], fontsize=9)
    axes[1].axis("off")

    # Panel 3: Marching cubes mesh (pyvista screenshot)
    mesh_raw = _pv_screenshot(state_ext.vertices, state_ext.faces)
    axes[2].imshow(mesh_raw)
    axes[2].set_title(titles[2], fontsize=9)
    axes[2].axis("off")

    # Panel 4: Smoothed mesh
    mesh_smooth = _pv_screenshot(state_smo.best_vertices, state_smo.faces)
    axes[3].imshow(mesh_smooth)
    axes[3].set_title(titles[3], fontsize=9)
    axes[3].axis("off")

    fig.patch.set_facecolor(BG_COLOR)
    fig.tight_layout(pad=1.0)
    fig.savefig(OUTPUT_DIR / "pipeline-stages.png", dpi=DPI, bbox_inches="tight",
                facecolor=BG_COLOR)
    plt.close(fig)


def gen_field_types() -> None:
    """Image 3: Density vs SDF 2x2 comparison grid (2D fields)."""
    import numpy as np
    from xeltofab.io import load_field
    from xeltofab.pipeline import process
    from xeltofab.state import PipelineParams, PipelineState

    # Load real density field
    state_density = load_field("data/examples/beams_2d_100x200_sample0.npy")
    result_density = process(state_density)

    # Generate synthetic 2D SDF: signed distance to two circles
    y, x = np.mgrid[0:100, 0:200].astype(np.float64)
    d1 = np.sqrt((x - 70) ** 2 + (y - 50) ** 2) - 25
    d2 = np.sqrt((x - 140) ** 2 + (y - 50) ** 2) - 20
    sdf_field = np.minimum(d1, d2)  # union of two circles

    params_sdf = PipelineParams(field_type="sdf")
    state_sdf = PipelineState(field=sdf_field, params=params_sdf)
    result_sdf = process(state_sdf)

    vmax_sdf = np.abs(sdf_field).max()

    fig, axes = plt.subplots(2, 2, figsize=(10, 5.5))

    # Top-left: Density field heatmap
    im0 = axes[0, 0].imshow(state_density.field, cmap="YlOrRd", origin="lower",
                             vmin=0, vmax=1)
    axes[0, 0].set_title("Density Field (0\u21921)", fontsize=10)
    axes[0, 0].axis("off")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # Top-right: SDF field heatmap
    im1 = axes[0, 1].imshow(sdf_field, cmap="RdBu", origin="lower",
                             vmin=-vmax_sdf, vmax=vmax_sdf)
    axes[0, 1].set_title("SDF Field (signed distance)", fontsize=10)
    axes[0, 1].axis("off")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Bottom-left: Same density colormap + extracted contours at threshold=0.5
    im2 = axes[1, 0].imshow(state_density.field, cmap="YlOrRd", origin="lower",
                             vmin=0, vmax=1)
    if result_density.contours:
        for contour in result_density.contours:
            axes[1, 0].plot(contour[:, 1], contour[:, 0], color="#1a1a1a",
                            linewidth=2.0)
    axes[1, 0].set_title("Extracted at threshold=0.5", fontsize=10)
    axes[1, 0].axis("off")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Bottom-right: Same SDF colormap + extracted contours at level=0
    im3 = axes[1, 1].imshow(sdf_field, cmap="RdBu", origin="lower",
                             vmin=-vmax_sdf, vmax=vmax_sdf)
    if result_sdf.contours:
        for contour in result_sdf.contours:
            axes[1, 1].plot(contour[:, 1], contour[:, 0], color="#1a1a1a",
                            linewidth=2.0)
    axes[1, 1].set_title("Extracted at level=0.0", fontsize=10)
    axes[1, 1].axis("off")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.patch.set_facecolor(BG_COLOR)
    fig.tight_layout(pad=1.5)
    fig.savefig(OUTPUT_DIR / "field-types-comparison.png", dpi=DPI, bbox_inches="tight",
                facecolor=BG_COLOR)
    plt.close(fig)


def gen_parameter_sensitivity() -> None:
    """Image 4: 3x3 parameter sensitivity grid."""
    import numpy as np
    from xeltofab.io import load_field
    from xeltofab.preprocess import preprocess
    from xeltofab.extract import extract
    from xeltofab.smooth import smooth
    from xeltofab.state import PipelineParams

    # Load data — thermoelastic is coarser (16^3), making smoothing diffs visible
    state_2d = load_field("data/examples/beams_2d_100x200_sample0.npy")
    state_3d = load_field("data/examples/thermoelastic_3d_16x16x16_sample0.npy")

    fig, axes = plt.subplots(3, 3, figsize=(10, 7.5))
    fig.patch.set_facecolor(BG_COLOR)

    # Row 1: threshold variation (2D contours on density field)
    thresholds = [0.3, 0.5, 0.7]
    for col, thresh in enumerate(thresholds):
        params = PipelineParams(threshold=thresh)
        st = state_2d.model_copy(update={"params": params})
        st = preprocess(st)
        st = extract(st)
        ax = axes[0, col]
        ax.imshow(state_2d.field, cmap="gray_r", origin="lower", alpha=0.4)
        if st.contours:
            for c in st.contours:
                ax.plot(c[:, 1], c[:, 0], "b-", linewidth=1.2)
        ax.set_title(f"threshold={thresh}", fontsize=9)
        ax.axis("off")
    axes[0, 0].set_ylabel("Threshold", fontsize=10, fontweight="bold")

    # Row 2: sigma variation (2D — show contours after preprocess + extract)
    sigmas = [0.0, 1.0, 2.0]
    for col, sigma in enumerate(sigmas):
        params = PipelineParams(smooth_sigma=sigma)
        st = state_2d.model_copy(update={"params": params})
        st = preprocess(st)
        st = extract(st)
        ax = axes[1, col]
        ax.imshow(state_2d.field, cmap="gray_r", origin="lower", alpha=0.4)
        if st.contours:
            for c in st.contours:
                ax.plot(c[:, 1], c[:, 0], "b-", linewidth=1.2)
        ax.set_title(f"smooth_sigma={sigma}", fontsize=9)
        ax.axis("off")
    axes[1, 0].set_ylabel("Sigma", fontsize=10, fontweight="bold")

    # Row 3: taubin_iterations variation (3D mesh screenshots)
    iterations = [0, 10, 50]
    state_3d_pre = preprocess(state_3d)
    state_3d_ext = extract(state_3d_pre)
    for col, iters in enumerate(iterations):
        if iters == 0:
            verts = state_3d_ext.vertices
        else:
            params = PipelineParams(taubin_iterations=iters)
            st = state_3d_ext.model_copy(update={"params": params})
            st = smooth(st)
            verts = st.best_vertices
        img = _pv_screenshot(verts, state_3d_ext.faces)
        ax = axes[2, col]
        ax.imshow(img)
        ax.set_title(f"taubin_iterations={iters}", fontsize=9)
        ax.axis("off")
    axes[2, 0].set_ylabel("Taubin", fontsize=10, fontweight="bold")

    fig.tight_layout(pad=1.5)
    fig.savefig(OUTPUT_DIR / "parameter-sensitivity.png", dpi=DPI, bbox_inches="tight",
                facecolor=BG_COLOR)
    plt.close(fig)


def gen_quality_metrics() -> None:
    """Image 5: Scaled Jacobian heatmap + histogram composite."""
    import numpy as np
    import pyvista as pv
    from xeltofab.io import load_field
    from xeltofab.pipeline import process
    from xeltofab.quality_plots import (  # internal APIs for reuse
        _build_pv_mesh,
        _compute_cell_metric,
        _compute_pass_rate,
        _HIGHER_IS_BETTER,
        _METRIC_LABELS,
        _THRESHOLDS,
    )

    # Process real TO data through full pipeline
    state = load_field("data/examples/heat_conduction_3d_51x51x51_sample0.npy")
    result = process(state)

    metric = "scaled_jacobian"
    threshold = _THRESHOLDS[metric]

    # Build mesh and compute metric
    pv_mesh = _build_pv_mesh(result)
    values = _compute_cell_metric(pv_mesh, metric)
    pass_pct = _compute_pass_rate(values, metric, threshold)

    # Left panel: pyvista heatmap screenshot
    pv_mesh.cell_data[metric] = values
    pl = pv.Plotter(off_screen=True, window_size=[800, 600])
    pl.add_mesh(
        pv_mesh, scalars=metric, cmap="RdYlGn",
        show_edges=True, edge_color="black", line_width=0.3,
        scalar_bar_args={"title": _METRIC_LABELS[metric]},
    )
    pl.camera_position = "iso"
    pl.set_background(BG_COLOR)
    heatmap_img = pl.screenshot(return_img=True)
    pl.close()

    # Composite: left = heatmap image, right = histogram
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4.5),
                                             gridspec_kw={"width_ratios": [1.2, 1]})

    ax_left.imshow(heatmap_img)
    ax_left.set_title("Per-Cell Scaled Jacobian", fontsize=10)
    ax_left.axis("off")

    # Right: histogram
    counts, bin_edges, patches = ax_right.hist(
        values, bins=50, edgecolor="black", linewidth=0.5, alpha=0.8,
        color="#3182BD",
    )
    ax_right.axvline(threshold, color="#D0021B", linestyle="--", linewidth=2,
                     label=f"FEA threshold ({threshold})")

    # Extend y-axis to make room for annotation above bars
    y_max = counts.max()
    ax_right.set_ylim(0, y_max * 1.35)

    direction = ">=" if _HIGHER_IS_BETTER[metric] else "<="
    ax_right.annotate(
        f"{pass_pct:.1f}% pass ({direction} {threshold})\n"
        f"mean={np.mean(values):.2f}, median={np.median(values):.2f}",
        xy=(0.97, 0.97), xycoords="axes fraction", ha="right", va="top",
        fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
    )
    ax_right.set_xlabel(_METRIC_LABELS[metric], fontsize=10)
    ax_right.set_ylabel("Number of Cells", fontsize=10)
    ax_right.set_title("Distribution", fontsize=10)
    ax_right.legend(loc="upper left", fontsize=8)

    fig.patch.set_facecolor(BG_COLOR)
    fig.tight_layout(pad=1.5)
    fig.savefig(OUTPUT_DIR / "quality-jacobian.png", dpi=DPI, bbox_inches="tight",
                facecolor=BG_COLOR)
    plt.close(fig)


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
