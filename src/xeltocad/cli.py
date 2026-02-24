"""CLI entrypoint for xelToCAD."""

from __future__ import annotations

from pathlib import Path

import click

from xeltocad.io import load_density, save_mesh
from xeltocad.pipeline import process
from xeltocad.state import PipelineParams
from xeltocad.viz import plot_comparison


@click.group()
def main() -> None:
    """xelToCAD — Topology optimization post-processing pipeline."""


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", "output_path", type=click.Path(path_type=Path), required=True)
@click.option("--threshold", type=float, default=0.5, help="Density threshold [0-1]")
@click.option("--sigma", type=float, default=1.0, help="Gaussian smoothing sigma")
@click.option("--viz", is_flag=True, help="Save a comparison visualization alongside the mesh")
def process_cmd(input_path: Path, output_path: Path, threshold: float, sigma: float, viz: bool) -> None:
    """Process a density field into a mesh."""
    params = PipelineParams(threshold=threshold, smooth_sigma=sigma)
    state = load_density(input_path, params=params)
    state = process(state)
    save_mesh(state, output_path)
    click.echo(f"Saved mesh to {output_path}")

    if viz:
        fig = plot_comparison(state)
        viz_path = output_path.with_suffix(".png")
        fig.savefig(viz_path, dpi=150)
        click.echo(f"Saved visualization to {viz_path}")


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", "output_path", type=click.Path(path_type=Path), default=None)
@click.option("--threshold", type=float, default=0.5, help="Density threshold [0-1]")
@click.option("--sigma", type=float, default=1.0, help="Gaussian smoothing sigma")
def viz(input_path: Path, output_path: Path | None, threshold: float, sigma: float) -> None:
    """Visualize a density field and its extraction result."""
    params = PipelineParams(threshold=threshold, smooth_sigma=sigma)
    state = load_density(input_path, params=params)
    state = process(state)
    fig = plot_comparison(state)

    if output_path:
        fig.savefig(output_path, dpi=150)
        click.echo(f"Saved visualization to {output_path}")
    else:
        import matplotlib.pyplot as plt

        plt.show()
