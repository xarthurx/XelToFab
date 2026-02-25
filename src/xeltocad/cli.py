"""CLI entrypoint for xelToCAD."""

from __future__ import annotations

from pathlib import Path

import click

from xeltocad.io import load_density, save_mesh
from xeltocad.loaders import get_supported_formats
from xeltocad.pipeline import process
from xeltocad.state import PipelineParams
from xeltocad.viz import plot_comparison


def _parse_shape(value: str) -> tuple[int, ...]:
    """Parse a shape string like '100x200' or '10x20x30' into a tuple."""
    parts = value.lower().split("x")
    return tuple(int(p) for p in parts)


@click.group()
def main() -> None:
    """xelToCAD — Topology optimization post-processing pipeline."""


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", "output_path", type=click.Path(path_type=Path), required=True)
@click.option("--threshold", type=float, default=0.5, help="Density threshold [0-1]")
@click.option("--sigma", type=float, default=1.0, help="Gaussian smoothing sigma")
@click.option("-f", "--field-name", default=None, help="Field/variable name to extract from input file")
@click.option("--shape", "shape_str", default=None, help="Grid shape for flat data, e.g. 100x200 or 10x20x30")
@click.option("--viz", is_flag=True, help="Save a comparison visualization alongside the mesh")
def process_cmd(
    input_path: Path,
    output_path: Path,
    threshold: float,
    sigma: float,
    field_name: str | None,
    shape_str: str | None,
    viz: bool,
) -> None:
    """Process a density field into a mesh."""
    params = PipelineParams(threshold=threshold, smooth_sigma=sigma)
    shape = _parse_shape(shape_str) if shape_str else None

    try:
        state = load_density(input_path, field_name=field_name, shape=shape, params=params)
    except (ValueError, ImportError) as e:
        raise click.ClickException(str(e)) from None

    import matplotlib.pyplot as plt

    state = process(state)
    try:
        save_mesh(state, output_path)
    except ValueError as e:
        raise click.ClickException(str(e)) from None
    click.echo(f"Saved mesh to {output_path}")

    if viz:
        fig = plot_comparison(state)
        viz_path = output_path.with_suffix(".png")
        fig.savefig(viz_path, dpi=150)
        plt.close(fig)
        click.echo(f"Saved visualization to {viz_path}")


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", "output_path", type=click.Path(path_type=Path), default=None)
@click.option("--threshold", type=float, default=0.5, help="Density threshold [0-1]")
@click.option("--sigma", type=float, default=1.0, help="Gaussian smoothing sigma")
@click.option("-f", "--field-name", default=None, help="Field/variable name to extract from input file")
@click.option("--shape", "shape_str", default=None, help="Grid shape for flat data, e.g. 100x200 or 10x20x30")
def viz(
    input_path: Path,
    output_path: Path | None,
    threshold: float,
    sigma: float,
    field_name: str | None,
    shape_str: str | None,
) -> None:
    """Visualize a density field and its extraction result."""
    params = PipelineParams(threshold=threshold, smooth_sigma=sigma)
    shape = _parse_shape(shape_str) if shape_str else None

    try:
        state = load_density(input_path, field_name=field_name, shape=shape, params=params)
    except (ValueError, ImportError) as e:
        raise click.ClickException(str(e)) from None

    state = process(state)
    fig = plot_comparison(state)

    if output_path:
        import matplotlib.pyplot as plt

        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        click.echo(f"Saved visualization to {output_path}")
    else:
        import matplotlib.pyplot as plt

        plt.show()


@main.command()
def formats() -> None:
    """List supported input formats and their availability."""
    fmt_list = get_supported_formats()
    click.echo(f"{'Format':<10} {'Extensions':<20} {'Status':<12} {'Install'}")
    click.echo("-" * 65)
    for f in fmt_list:
        exts = ", ".join(f["extensions"])
        status = "available" if f["available"] else "missing"
        click.echo(f"{f['name']:<10} {exts:<20} {status:<12} {f['install_hint']}")
