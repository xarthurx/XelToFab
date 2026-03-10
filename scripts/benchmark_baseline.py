"""Benchmark baseline: capture mesh quality metrics and visualizations.

Usage:
    uv run python scripts/benchmark_baseline.py [--output-dir benchmarks/baseline]

Generates mesh files, comparison plots, 3D renders, and a metrics summary
for each model in the registry. Re-run with a different --output-dir after
pipeline changes to compare before/after quality.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from xeltofab.io import load_field, save_mesh
from xeltofab.pipeline import process
from xeltofab.quality import compute_quality
from xeltofab.state import PipelineParams, PipelineState
from xeltofab.field_plots import plot_comparison
from xeltofab.quality_plots import plot_metric_overview, plot_quality_overview

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkModel:
    name: str
    field_type: Literal["density", "sdf"] = "density"
    path: str | None = None  # None = synthetic (generated inline)
    generator: Literal["sphere", "sdf_sphere"] | None = None


MODELS: list[BenchmarkModel] = [
    # EngiBench 3D
    BenchmarkModel("heat_cond_51_s0", path="data/examples/heat_conduction_3d_51x51x51_sample0.npy"),
    BenchmarkModel("heat_cond_51_s1", path="data/examples/heat_conduction_3d_51x51x51_sample1.npy"),
    BenchmarkModel("thermoelastic_16_s0", path="data/examples/thermoelastic_3d_16x16x16_sample0.npy"),
    # Corner-Based TO 3D
    BenchmarkModel("corner_3d_vf50_s0", path="data/examples/corner_3d_20x20x20_vf50_sample0.npy"),
    BenchmarkModel("corner_3d_vf30_s0", path="data/examples/corner_3d_20x20x20_vf30_sample0.npy"),
    # EngiBench 2D
    BenchmarkModel("beam_2d_100x200_s0", path="data/examples/beams_2d_100x200_sample0.npy"),
    # Synthetic
    BenchmarkModel("synthetic_sphere", generator="sphere"),
    BenchmarkModel("synthetic_sphere_sdf", field_type="sdf", generator="sdf_sphere"),
]


def _generate_synthetic(generator: str) -> np.ndarray:
    """Generate synthetic density/SDF fields for controlled benchmarking."""
    if generator == "sphere":
        z, y, x = np.mgrid[-1:1:30j, -1:1:30j, -1:1:30j]
        return np.clip(1.0 - np.sqrt(x**2 + y**2 + z**2) / 0.5, 0, 1)
    if generator == "sdf_sphere":
        z, y, x = np.mgrid[-1:1:30j, -1:1:30j, -1:1:30j]
        return np.sqrt(x**2 + y**2 + z**2) - 0.5
    raise ValueError(f"Unknown generator: {generator}")


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _to_pyvista(state: PipelineState):
    """Build a pyvista PolyData from pipeline state, or None if unavailable."""
    vertices = state.best_vertices
    if vertices is None or state.faces is None:
        return None
    try:
        import pyvista as pv

        faces_pv = np.column_stack([np.full(len(state.faces), 3), state.faces]).ravel()
        return pv.PolyData(vertices.astype(np.float64), faces_pv)
    except ImportError:
        return None


def compute_metrics(state: PipelineState, elapsed: float, pv_mesh=None) -> dict:
    """Compute mesh quality metrics from a processed pipeline state."""
    metrics = compute_quality(state)
    metrics["input_shape"] = list(state.field.shape)
    metrics["field_type"] = state.params.field_type
    metrics["direct_extraction"] = state.params.direct_extraction
    metrics["processing_time_s"] = round(elapsed, 4)
    return metrics


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_comparison_plot(state: PipelineState, output_path: Path) -> None:
    """Save a matplotlib comparison plot (field vs result)."""
    fig = plot_comparison(state)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_mesh_render(pv_mesh, output_path: Path) -> None:
    """Save a pyvista 3D mesh render (off-screen)."""
    if pv_mesh is None:
        return

    try:
        import pyvista as pv

        pl = pv.Plotter(off_screen=True, window_size=[1024, 768])
        pl.add_mesh(pv_mesh, color="steelblue", show_edges=True, edge_color="black", line_width=0.3)
        pl.camera_position = "iso"
        pl.screenshot(str(output_path))
        pl.close()
    except Exception as e:
        print(f"  WARNING: 3D render failed: {e}")


def save_quality_heatmap(state: PipelineState, output_path: Path) -> None:
    """Save a 1x3 quality heatmap overview (off-screen)."""
    try:
        pl = plot_quality_overview(state)
        pl.screenshot(str(output_path))
        pl.close()
    except Exception as e:
        print(f"  WARNING: Quality heatmap failed: {e}")


def save_quality_histograms(state: PipelineState, output_path: Path) -> None:
    """Save a 1x3 metric histogram overview."""
    try:
        fig = plot_metric_overview(state)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"  WARNING: Quality histograms failed: {e}")


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def write_summary(all_metrics: dict, output_dir: Path) -> None:
    """Write a markdown summary table from collected metrics."""
    lines = [
        "# Benchmark Summary",
        "",
        f"Generated by `scripts/benchmark_baseline.py` into `{output_dir}`.",
        "",
        "## 3D Models",
        "",
        "| Model | Verts | Faces | Watertight | Volume | AR (mean) | Angle (min) | Jac (min) | Time |",
        "|-------|-------|-------|------------|--------|-----------|-------------|-----------|------|",
    ]

    for name, m in all_metrics.items():
        if m.get("ndim") != 3:
            continue
        verts = m.get("num_vertices", "—")
        faces = m.get("num_faces", "—")
        watertight = "yes" if m.get("is_watertight") else "no"
        volume = f"{m['volume']:.4f}" if "volume" in m else "—"
        ar = f"{m['aspect_ratio']['mean']:.2f}" if "aspect_ratio" in m else "—"
        ma = f"{m['min_angle']['min']:.1f} deg" if "min_angle" in m else "—"
        sj = f"{m['scaled_jacobian']['min']:.3f}" if "scaled_jacobian" in m else "—"
        t = f"{m['processing_time_s']:.3f}"
        lines.append(f"| {name} | {verts} | {faces} | {watertight} | {volume} | {ar} | {ma} | {sj} | {t} |")

    lines += [
        "",
        "## 2D Models",
        "",
        "| Model | Contours | Total Points | Volume Fraction | Time (s) |",
        "|-------|----------|--------------|-----------------|----------|",
    ]

    for name, m in all_metrics.items():
        if m.get("ndim") != 2:
            continue
        nc = m.get("num_contours", "—")
        tp = m.get("total_contour_points", "—")
        vf = f"{m['volume_fraction']:.4f}" if "volume_fraction" in m else "—"
        t = f"{m['processing_time_s']:.3f}"
        lines.append(f"| {name} | {nc} | {tp} | {vf} | {t} |")

    lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run pipeline benchmark and capture quality baseline.")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/baseline"))
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: dict[str, dict] = {}

    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"Processing: {model.name}")
        print(f"{'='*60}")

        # Load
        params = PipelineParams(field_type=model.field_type)
        if model.path:
            state = load_field(model.path, params=params)
        else:
            field = _generate_synthetic(model.generator)
            state = PipelineState(field=field, params=params)

        print(f"  Input shape: {state.field.shape}, field_type: {model.field_type}")

        # Process
        t0 = time.perf_counter()
        result = process(state)
        elapsed = time.perf_counter() - t0
        print(f"  Processed in {elapsed:.3f}s")

        # Build pyvista mesh once (reused for metrics + render)
        pv_mesh = _to_pyvista(result) if result.ndim == 3 else None

        # Metrics
        metrics = compute_metrics(result, elapsed, pv_mesh=pv_mesh)
        all_metrics[model.name] = metrics
        if "num_vertices" in metrics:
            print(f"  Mesh: {metrics['num_vertices']} verts, {metrics['num_faces']} faces")
        if "aspect_ratio" in metrics:
            ar = metrics["aspect_ratio"]
            print(f"  Aspect ratio: mean={ar['mean']:.2f}, max={ar['max']:.2f}")
        if "min_angle" in metrics:
            print(f"  Min angle: min={metrics['min_angle']['min']:.1f} deg")

        # Save mesh (3D only)
        if result.ndim == 3 and result.vertices is not None:
            mesh_path = output_dir / f"{model.name}.stl"
            save_mesh(result, mesh_path)
            print(f"  Saved mesh: {mesh_path}")

        # Save comparison plot
        comp_path = output_dir / f"{model.name}_comparison.png"
        save_comparison_plot(result, comp_path)
        print(f"  Saved comparison: {comp_path}")

        # Save 3D render (3D only)
        if pv_mesh is not None:
            render_path = output_dir / f"{model.name}_mesh.png"
            save_mesh_render(pv_mesh, render_path)
            print(f"  Saved 3D render: {render_path}")

        # Save quality heatmaps (3D only)
        if result.ndim == 3 and result.vertices is not None:
            quality_path = output_dir / f"{model.name}_quality.png"
            save_quality_heatmap(result, quality_path)
            print(f"  Saved quality heatmap: {quality_path}")

            hist_path = output_dir / f"{model.name}_histograms.png"
            save_quality_histograms(result, hist_path)
            print(f"  Saved histograms: {hist_path}")

    # Write metrics JSON
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2))
    print(f"\nMetrics saved to {metrics_path}")

    # Write summary
    write_summary(all_metrics, output_dir)
    print(f"Summary saved to {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
