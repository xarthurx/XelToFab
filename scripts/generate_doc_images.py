"""Generate documentation images for website guide and getting-started pages.

Usage:
    uv run python scripts/generate_doc_images.py [--only NAME]

Valid --only values: pipeline_diagram, pipeline_stages, field_types,
    parameter_sensitivity, quality_metrics, hero_overview,
    quickstart_2d, quickstart_smoothing
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Shared constants
OUTPUT_DIR = Path("website/public/images/guides")
OUTPUT_DIR_GS = Path("website/public/images/getting-started")
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
    OUTPUT_DIR_GS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pv_screenshot(vertices, faces, color="#88BDE6", zoom=1.0):
    """Render a mesh off-screen and return the image as a numpy array."""
    import numpy as np
    import pyvista as pv

    faces_pv = np.column_stack([np.full(len(faces), 3), faces]).ravel()
    pv_mesh = pv.PolyData(vertices.astype(np.float64), faces_pv)

    pl = pv.Plotter(off_screen=True, window_size=[512, 512])
    pl.add_mesh(pv_mesh, color=color, show_edges=True, edge_color="black", line_width=0.3)
    pl.camera_position = "iso"
    if zoom != 1.0:
        pl.camera.zoom(zoom)
    pl.set_background(BG_COLOR)
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


# ---------------------------------------------------------------------------
# Image generators (one per function)
# ---------------------------------------------------------------------------


def gen_pipeline_diagram() -> None:
    """Image 1: 3-row pipeline flow diagram with blue palette.

    Row 1: Field terminal (top-left)
    Row 2: 6 pipeline stages with horizontal arrows
    Row 3: Mesh terminal (bottom-right)
    Vertical arrows connect Field→row2 and row2→Mesh.
    Optional stages have a dashed outline offset around the solid box.
    """
    stages = list(STAGE_COLORS.keys())

    # Layout constants
    gap = 1.8        # horizontal gap between stage centers
    row_gap = 2.0    # vertical gap between row centers
    box_hw = 0.55    # half-width
    box_hh = 0.3     # half-height
    dash_gap = 0.12  # offset between solid box and dashed outline

    # Stage positions (row 2, y=0)
    stage_x = [i * gap for i in range(len(stages))]
    total_w = stage_x[-1]

    # Terminal positions
    field_x, field_y = stage_x[0], row_gap
    mesh_x, mesh_y = stage_x[-1], -row_gap

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(-1.0, total_w + 1.3)
    ax.set_ylim(-row_gap - 0.9, row_gap + 0.7)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(BG_COLOR)

    param_annotations = {
        "Preprocess": "threshold\nsmooth_sigma",
        "Extract": "extraction_level",
        "Smooth": "taubin_iterations\nsmoothing_method",
        "Repair": "repair",
        "Remesh": "edge_length\nremesh_iterations",
        "Decimate": "decimate_ratio\naggressiveness",
    }
    optional_stages = {"Preprocess", "Repair", "Remesh", "Decimate"}

    def _draw_box(x, y, label, is_terminal=False):
        if is_terminal:
            fc, ec, tc = "#E8E8E8", "#999999", "#333333"
        else:
            fc = STAGE_COLORS[label]
            ec = "#333333"
            tc = _STAGE_TEXT[label]

        # Solid box
        ax.add_patch(matplotlib.patches.FancyBboxPatch(
            (x - box_hw, y - box_hh), box_hw * 2, box_hh * 2,
            boxstyle="round,pad=0.08",
            facecolor=fc, edgecolor=ec, linewidth=1.2,
        ))
        ax.text(x, y, label, ha="center", va="center",
                fontsize=9 if is_terminal else 8.5, fontweight="bold", color=tc)

        # Dashed outline for optional stages (offset around solid box)
        if label in optional_stages:
            d = dash_gap
            ax.add_patch(matplotlib.patches.FancyBboxPatch(
                (x - box_hw - d, y - box_hh - d),
                (box_hw + d) * 2, (box_hh + d) * 2,
                boxstyle="round,pad=0.08",
                facecolor="none", edgecolor="#888888",
                linewidth=1.0, linestyle="--",
            ))

    # --- Row 1: Field terminal ---
    _draw_box(field_x, field_y, "Field", is_terminal=True)

    # --- Row 2: Pipeline stages ---
    for i, label in enumerate(stages):
        _draw_box(stage_x[i], 0, label)

        # Parameter annotation below stage
        if label in param_annotations:
            ax.text(stage_x[i], -box_hh - dash_gap - 0.15, param_annotations[label],
                    ha="center", va="top", fontsize=7, color="#666666",
                    fontstyle="italic")

    # --- Row 3: Mesh terminal ---
    _draw_box(mesh_x, mesh_y, "Mesh", is_terminal=True)

    # --- Arrows ---
    arrow_kw = dict(arrowstyle="-|>", color="#555555", lw=1.3)
    arrow_gap = 0.1  # clearance between arrow tip and nearest border

    def _box_edge(label, is_terminal=False):
        """Return half-extent from center to outermost border (including dashed)."""
        hw = box_hw
        if not is_terminal and label in optional_stages:
            hw += dash_gap
        return hw

    # Field → Preprocess (vertical)
    ax.annotate("",
                xy=(field_x, _box_edge("Preprocess") + arrow_gap),
                xytext=(field_x, field_y - box_hh - arrow_gap),
                arrowprops=arrow_kw)

    # Horizontal arrows between stages
    for i in range(len(stages) - 1):
        src, dst = stages[i], stages[i + 1]
        x_start = stage_x[i] + _box_edge(src) + arrow_gap
        x_end = stage_x[i + 1] - _box_edge(dst) - arrow_gap
        ax.annotate("", xy=(x_end, 0), xytext=(x_start, 0),
                    arrowprops=arrow_kw)

    # Decimate → Mesh (straight vertical)
    ax.annotate("",
                xy=(mesh_x, mesh_y + box_hh + arrow_gap),
                xytext=(mesh_x, -(_box_edge("Decimate") + arrow_gap)),
                arrowprops=arrow_kw)

    # --- Legend (bottom-left) ---
    leg_x = -0.5
    leg_y = -row_gap - 0.1
    ax.add_patch(matplotlib.patches.FancyBboxPatch(
        (leg_x, leg_y), 0.7, 0.35,
        boxstyle="round,pad=0.05",
        facecolor="none", edgecolor="#888888", linewidth=1.0, linestyle="--",
    ))
    ax.text(leg_x + 0.85, leg_y + 0.175, "= optional stage",
            ha="left", va="center", fontsize=7, color="#666666")

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

    fig, axes = plt.subplots(1, 4, figsize=(12, 4.0))
    # Main titles (line 1) and subtitles (line 2, parenthetical)
    main_titles = ["Raw Field", "After Threshold", "Marching Cubes", "After Smoothing"]
    subtitles = ["(center slice)", "(binary)", "(raw mesh)", ""]

    # Panel 1: Center slice of raw density field
    field = state.field
    mid = field.shape[0] // 2
    axes[0].imshow(field[mid], cmap="viridis", origin="lower")
    axes[0].axis("off")

    # Panel 2: Center slice of binary field
    assert state_pre.binary is not None
    axes[1].imshow(state_pre.binary[mid], cmap="gray", origin="lower")
    axes[1].axis("off")

    # Panel 3: Marching cubes mesh (pyvista screenshot)
    mesh_raw = _pv_screenshot(state_ext.vertices, state_ext.faces)
    axes[2].imshow(mesh_raw)
    axes[2].axis("off")

    # Panel 4: Smoothed mesh
    mesh_smooth = _pv_screenshot(state_smo.best_vertices, state_smo.faces)
    axes[3].imshow(mesh_smooth)
    axes[3].axis("off")

    fig.patch.set_facecolor(BG_COLOR)
    fig.tight_layout(pad=0.5)

    # Place titles at uniform y — main title top-aligned, subtitle tight below
    for ax, main, sub in zip(axes, main_titles, subtitles, strict=True):
        bbox = ax.get_position()
        cx = bbox.x0 + bbox.width / 2
        y_top = bbox.y1 + 0.1
        fig.text(cx, y_top, main, ha="center", va="top", fontsize=9,
                 fontweight="bold")
        if sub:
            fig.text(cx, y_top - 0.04, sub, ha="center", va="top",
                     fontsize=8, color="#666666")

    fig.savefig(OUTPUT_DIR / "pipeline-stages.png", dpi=DPI, bbox_inches="tight",
                facecolor=BG_COLOR)
    plt.close(fig)


def gen_field_types() -> None:
    """Image 3: Density vs SDF 1x2 — field heatmap with contour overlay."""
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

    fig, (ax_den, ax_sdf) = plt.subplots(1, 2, figsize=(10, 3.2))

    # Left: Density field + extracted contours
    im0 = ax_den.imshow(state_density.field, cmap="YlOrRd", origin="lower",
                         vmin=0, vmax=1)
    if result_density.contours:
        for contour in result_density.contours:
            ax_den.plot(contour[:, 1], contour[:, 0], color="#1a1a1a",
                        linewidth=1.8)
    ax_den.set_title("Density field — extracted at threshold = 0.5", fontsize=9.5)
    ax_den.axis("off")
    fig.colorbar(im0, ax=ax_den, fraction=0.046, pad=0.04)

    # Right: SDF field + extracted contours
    im1 = ax_sdf.imshow(sdf_field, cmap="RdBu", origin="lower",
                         vmin=-vmax_sdf, vmax=vmax_sdf)
    if result_sdf.contours:
        for contour in result_sdf.contours:
            ax_sdf.plot(contour[:, 1], contour[:, 0], color="#1a1a1a",
                        linewidth=1.8)
    ax_sdf.set_title("SDF field — extracted at level = 0.0", fontsize=9.5)
    ax_sdf.axis("off")
    fig.colorbar(im1, ax=ax_sdf, fraction=0.046, pad=0.04)

    fig.patch.set_facecolor(BG_COLOR)
    fig.tight_layout(pad=1.5)
    fig.savefig(OUTPUT_DIR / "field-types-comparison.png", dpi=DPI, bbox_inches="tight",
                facecolor=BG_COLOR)
    plt.close(fig)


def gen_parameter_sensitivity() -> None:
    """Image 4: Three separate 1x3 strips for threshold, sigma, and taubin."""
    _gen_param_threshold()
    _gen_param_sigma()
    _gen_param_taubin()


def _gen_param_threshold() -> None:
    """Threshold variation: 1x3 strip showing contours at 0.3, 0.5, 0.7."""
    from xeltofab.io import load_field
    from xeltofab.preprocess import preprocess
    from xeltofab.extract import extract
    from xeltofab.state import PipelineParams

    state_2d = load_field("data/examples/beams_2d_100x200_sample0.npy")

    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    fig.patch.set_facecolor(BG_COLOR)

    thresholds = [0.3, 0.5, 0.7]
    for col, thresh in enumerate(thresholds):
        params = PipelineParams(threshold=thresh)
        st = state_2d.model_copy(update={"params": params})
        st = preprocess(st)
        st = extract(st)
        ax = axes[col]
        ax.imshow(state_2d.field, cmap="YlOrRd", origin="lower", vmin=0, vmax=1)
        if st.contours:
            for c in st.contours:
                ax.plot(c[:, 1], c[:, 0], color="#1a1a1a", linewidth=1.5)
        ax.set_title(f"threshold = {thresh}", fontsize=10)
        ax.axis("off")

    fig.tight_layout(pad=1.0)
    fig.savefig(OUTPUT_DIR / "param-threshold.png", dpi=DPI, bbox_inches="tight",
                facecolor=BG_COLOR)
    plt.close(fig)


def _gen_param_sigma() -> None:
    """Sigma variation: 1x3 strip showing contours at sigma 0, 1, 2."""
    from xeltofab.io import load_field
    from xeltofab.preprocess import preprocess
    from xeltofab.extract import extract
    from xeltofab.state import PipelineParams

    state_2d = load_field("data/examples/beams_2d_100x200_sample0.npy")

    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    fig.patch.set_facecolor(BG_COLOR)

    sigmas = [0.0, 1.0, 2.0]
    for col, sigma in enumerate(sigmas):
        params = PipelineParams(smooth_sigma=sigma)
        st = state_2d.model_copy(update={"params": params})
        st = preprocess(st)
        st = extract(st)
        ax = axes[col]
        ax.imshow(state_2d.field, cmap="YlOrRd", origin="lower", vmin=0, vmax=1)
        if st.contours:
            for c in st.contours:
                ax.plot(c[:, 1], c[:, 0], color="#1a1a1a", linewidth=1.5)
        ax.set_title(f"smooth_sigma = {sigma}", fontsize=10)
        ax.axis("off")

    fig.tight_layout(pad=1.0)
    fig.savefig(OUTPUT_DIR / "param-sigma.png", dpi=DPI, bbox_inches="tight",
                facecolor=BG_COLOR)
    plt.close(fig)


def _gen_param_taubin() -> None:
    """Taubin variation: 1x3 strip showing mesh at 0, 10, 50 iterations."""
    from xeltofab.io import load_field
    from xeltofab.preprocess import preprocess
    from xeltofab.extract import extract
    from xeltofab.smooth import smooth
    from xeltofab.state import PipelineParams

    # Thermoelastic — coarser mesh makes smoothing differences visible
    state_3d = load_field("data/examples/thermoelastic_3d_16x16x16_sample0.npy")
    state_3d_pre = preprocess(state_3d)
    state_3d_ext = extract(state_3d_pre)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    fig.patch.set_facecolor(BG_COLOR)

    iterations = [0, 10, 50]
    for col, iters in enumerate(iterations):
        if iters == 0:
            verts = state_3d_ext.vertices
        else:
            params = PipelineParams(taubin_iterations=iters)
            st = state_3d_ext.model_copy(update={"params": params})
            st = smooth(st)
            verts = st.best_vertices
        img = _pv_screenshot(verts, state_3d_ext.faces)
        ax = axes[col]
        ax.imshow(img)
        ax.set_title(f"taubin_iterations = {iters}", fontsize=10)
        ax.axis("off")

    fig.tight_layout(pad=1.0)
    fig.savefig(OUTPUT_DIR / "param-taubin.png", dpi=DPI, bbox_inches="tight",
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

    # Left panel: pyvista heatmap — no title in pyvista, we add it in matplotlib
    # Scalar bar at very bottom with small font to avoid overlapping the mesh
    pv_mesh.cell_data[metric] = values
    pl = pv.Plotter(off_screen=True, window_size=[800, 750])
    pl.add_mesh(
        pv_mesh, scalars=metric, cmap="RdYlGn",
        show_edges=True, edge_color="black", line_width=0.3,
        scalar_bar_args={
            "title": "",
            "label_font_size": 11,
            "position_x": 0.25,
            "position_y": 0.01,
            "width": 0.5,
            "height": 0.04,
            "vertical": False,
        },
    )
    pl.camera_position = "iso"
    pl.set_background(BG_COLOR)
    heatmap_img = pl.screenshot(return_img=True)
    pl.close()

    # Composite figure with generous spacing
    fig = plt.figure(figsize=(11, 5))
    fig.patch.set_facecolor(BG_COLOR)

    # Use gridspec for precise control
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.35)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    # Left: heatmap with title well above image
    ax_left.imshow(heatmap_img)
    ax_left.set_title("Per-Cell Scaled Jacobian", fontsize=11, pad=14)
    ax_left.axis("off")

    # Right: histogram
    counts, bin_edges, patches = ax_right.hist(
        values, bins=50, edgecolor="white", linewidth=0.3, alpha=0.9,
        color="#3182BD",
    )
    ax_right.axvline(threshold, color="#D0021B", linestyle="--", linewidth=2)

    # Extend y-axis generously so legend + stats sit above all bars
    y_max = counts.max()
    ax_right.set_ylim(0, y_max * 1.55)

    # Stats annotation — top right, well above bars
    direction = ">=" if _HIGHER_IS_BETTER[metric] else "<="
    ax_right.text(
        0.97, 0.97,
        f"{pass_pct:.1f}% pass ({direction} {threshold})\n"
        f"mean = {np.mean(values):.2f}   median = {np.median(values):.2f}",
        transform=ax_right.transAxes, ha="right", va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F0",
                  edgecolor="#CCCCCC", alpha=0.95),
    )

    # Threshold legend — top left, same vertical zone as stats
    ax_right.text(
        0.03, 0.97,
        f"- - FEA threshold ({threshold})",
        transform=ax_right.transAxes, ha="left", va="top",
        fontsize=8, color="#D0021B",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F0",
                  edgecolor="#CCCCCC", alpha=0.95),
    )

    ax_right.set_xlabel(_METRIC_LABELS[metric], fontsize=10)
    ax_right.set_ylabel("Number of Cells", fontsize=10)
    ax_right.set_title("Distribution", fontsize=11, pad=14)

    # Light grid for readability
    ax_right.yaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax_right.set_axisbelow(True)

    fig.savefig(OUTPUT_DIR / "quality-jacobian.png", dpi=DPI, bbox_inches="tight",
                facecolor=BG_COLOR)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tier 2 — Getting Started page images
# ---------------------------------------------------------------------------


def gen_hero_overview() -> None:
    """Tier 2 — Hero image for index: density field → XelToFab → triangle mesh."""
    from xeltofab.io import load_field
    from xeltofab.pipeline import process

    # Use heat_conduction model — higher res than corner (used in Tier 1 pipeline-stages)
    state = load_field("data/examples/heat_conduction_3d_51x51x51_sample0.npy")
    result = process(state)

    # Left: mid-slice of raw density field
    field = state.field
    mid = field.shape[0] // 2

    # Right: final smoothed mesh via pyvista
    mesh_img = _pv_screenshot(result.best_vertices, result.faces, zoom=1.4)

    # Use gridspec: left panel, center arrow gap, right panel
    fig = plt.figure(figsize=(10, 3.5))
    fig.patch.set_facecolor(BG_COLOR)
    gs = fig.add_gridspec(1, 3, width_ratios=[0.9, 0.22, 1.1], wspace=0.02)
    ax_field = fig.add_subplot(gs[0, 0])
    ax_arrow = fig.add_subplot(gs[0, 1])
    ax_mesh = fig.add_subplot(gs[0, 2])

    # Left panel: density field slice — no colorbar for a clean hero image
    ax_field.imshow(field[mid], cmap="YlOrRd", origin="lower", vmin=0, vmax=1)
    ax_field.set_title("Density Field", fontsize=11, fontweight="bold", color="#2B2B2B")
    ax_field.axis("off")

    # Center: horizontal arrow with label — minimal, no box
    ax_arrow.set_xlim(0, 1)
    ax_arrow.set_ylim(0, 1)
    ax_arrow.axis("off")
    ax_arrow.text(0.6, 0.50, "XelToFab", ha="center", va="bottom",
                  fontsize=9, fontweight="bold", color="#1B3A5C",
                  fontstyle="italic")
    ax_arrow.annotate(
        "", xy=(1.1, 0.46), xytext=(0.1, 0.46),
        arrowprops=dict(arrowstyle="-|>", color="#1B3A5C", lw=1.6,
                        mutation_scale=13),
        annotation_clip=False,
    )

    # Right panel: final mesh
    ax_mesh.imshow(mesh_img)
    ax_mesh.set_title("Triangle Mesh", fontsize=11, fontweight="bold", color="#2B2B2B")
    ax_mesh.axis("off")

    fig.savefig(OUTPUT_DIR_GS / "hero-overview.png", dpi=DPI, bbox_inches="tight",
                facecolor=BG_COLOR)
    plt.close(fig)


def gen_quickstart_2d() -> None:
    """Tier 2 — 2D comparison for quick-start: input field vs extracted contours."""
    from xeltofab.io import load_field
    from xeltofab.pipeline import process

    state = load_field("data/examples/beams_2d_100x200_sample0.npy")
    result = process(state)

    fig, (ax_in, ax_out) = plt.subplots(1, 2, figsize=(10, 3.2))
    fig.patch.set_facecolor(BG_COLOR)

    # Left: raw input field
    im = ax_in.imshow(state.field, cmap="YlOrRd", origin="lower", vmin=0, vmax=1)
    ax_in.set_title("Input Field", fontsize=11, fontweight="bold")
    ax_in.axis("off")
    fig.colorbar(im, ax=ax_in, fraction=0.046, pad=0.04)

    # Right: field with extracted contours overlaid
    ax_out.imshow(state.field, cmap="YlOrRd", origin="lower", vmin=0, vmax=1)
    if result.contours:
        for contour in result.contours:
            ax_out.plot(contour[:, 1], contour[:, 0], color="#1a1a1a", linewidth=1.5)
    ax_out.set_title("Extracted Contours", fontsize=11, fontweight="bold")
    ax_out.axis("off")

    fig.tight_layout(pad=1.5)
    fig.savefig(OUTPUT_DIR_GS / "quick-start-2d.png", dpi=DPI, bbox_inches="tight",
                facecolor=BG_COLOR)
    plt.close(fig)


def gen_quickstart_smoothing() -> None:
    """Tier 2 — Taubin vs bilateral smoothing for quick-start."""
    from xeltofab.io import load_field
    from xeltofab.preprocess import preprocess
    from xeltofab.extract import extract
    from xeltofab.smooth import smooth
    from xeltofab.state import PipelineParams

    # Corner model — sharp 90-degree features make difference visible
    state = load_field("data/examples/corner_3d_20x20x20_vf50_sample0.npy")
    state_pre = preprocess(state)
    state_ext = extract(state_pre)

    # Taubin smoothing (default)
    params_taubin = PipelineParams(taubin_iterations=30)
    st_taubin = state_ext.model_copy(update={"params": params_taubin})
    st_taubin = smooth(st_taubin)
    img_taubin = _pv_screenshot(st_taubin.best_vertices, st_taubin.faces)

    # Bilateral smoothing
    params_bilateral = PipelineParams(smoothing_method="bilateral")
    st_bilateral = state_ext.model_copy(update={"params": params_bilateral})
    st_bilateral = smooth(st_bilateral)
    img_bilateral = _pv_screenshot(st_bilateral.best_vertices, st_bilateral.faces)

    fig, (ax_tau, ax_bil) = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.patch.set_facecolor(BG_COLOR)

    ax_tau.imshow(img_taubin)
    ax_tau.set_title("Taubin Smoothing", fontsize=11, fontweight="bold")
    ax_tau.axis("off")

    ax_bil.imshow(img_bilateral)
    ax_bil.set_title("Bilateral Smoothing", fontsize=11, fontweight="bold")
    ax_bil.axis("off")

    fig.tight_layout(pad=1.0)
    fig.savefig(OUTPUT_DIR_GS / "quick-start-smoothing.png", dpi=DPI,
                bbox_inches="tight", facecolor=BG_COLOR)
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
    "hero_overview": gen_hero_overview,
    "quickstart_2d": gen_quickstart_2d,
    "quickstart_smoothing": gen_quickstart_smoothing,
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
