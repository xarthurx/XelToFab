"""CLI entrypoint for XelToFab."""

from __future__ import annotations

from pathlib import Path

import click

from xeltofab.io import load_density, save_mesh
from xeltofab.loaders import get_supported_formats
from xeltofab.pipeline import process
from xeltofab.state import PipelineParams
from xeltofab.viz import plot_comparison


def _parse_shape(value: str) -> tuple[int, ...]:
    """Parse a shape string like '100x200' or '10x20x30' into a tuple."""
    parts = value.lower().split("x")
    return tuple(int(p) for p in parts)


def _build_params(
    ctx: click.Context,
    threshold: float,
    sigma: float,
    field_type: str,
    direct: bool,
    no_repair: bool,
    no_remesh: bool,
) -> PipelineParams:
    """Build PipelineParams, only passing values explicitly set by the user.

    This preserves PipelineParams smart defaults (e.g., SDF auto-enables
    direct extraction and disables Gaussian smoothing).
    """
    kwargs: dict = {"threshold": threshold, "field_type": field_type}
    source = click.core.ParameterSource.COMMANDLINE
    if ctx.get_parameter_source("sigma") == source:
        kwargs["smooth_sigma"] = sigma
    if ctx.get_parameter_source("direct") == source:
        kwargs["direct_extraction"] = direct
    if no_repair:
        kwargs["repair"] = False
    if no_remesh:
        kwargs["remesh"] = False
    return PipelineParams(**kwargs)


@click.group()
def main() -> None:
    """XelToFab — Topology optimization post-processing pipeline."""


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", "output_path", type=click.Path(path_type=Path), required=True)
@click.option("--threshold", type=float, default=0.5, help="Density threshold [0-1]")
@click.option("--sigma", type=float, default=1.0, help="Gaussian smoothing sigma")
@click.option("-f", "--field-name", default=None, help="Field/variable name to extract from input file")
@click.option("--shape", "shape_str", default=None, help="Grid shape for flat data, e.g. 100x200 or 10x20x30")
@click.option("--field-type", type=click.Choice(["density", "sdf"]), default="density", help="Input field type")
@click.option("--direct", is_flag=True, help="Direct extraction from continuous field (skip preprocessing)")
@click.option("--no-repair", is_flag=True, help="Disable watertight mesh repair")
@click.option("--no-remesh", is_flag=True, help="Disable isotropic remeshing")
@click.option("--viz", is_flag=True, help="Save a comparison visualization alongside the mesh")
@click.pass_context
def process_cmd(
    ctx: click.Context,
    input_path: Path,
    output_path: Path,
    threshold: float,
    sigma: float,
    field_name: str | None,
    shape_str: str | None,
    field_type: str,
    direct: bool,
    no_repair: bool,
    no_remesh: bool,
    viz: bool,
) -> None:
    """Process a density field into a mesh."""
    params = _build_params(ctx, threshold, sigma, field_type, direct, no_repair, no_remesh)
    shape = _parse_shape(shape_str) if shape_str else None

    try:
        state = load_density(input_path, field_name=field_name, shape=shape, params=params)
    except (ValueError, KeyError, ImportError) as e:
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
@click.option("--field-type", type=click.Choice(["density", "sdf"]), default="density", help="Input field type")
@click.option("--direct", is_flag=True, help="Direct extraction from continuous field (skip preprocessing)")
@click.option("--no-repair", is_flag=True, help="Disable watertight mesh repair")
@click.option("--no-remesh", is_flag=True, help="Disable isotropic remeshing")
@click.pass_context
def viz(
    ctx: click.Context,
    input_path: Path,
    output_path: Path | None,
    threshold: float,
    sigma: float,
    field_name: str | None,
    shape_str: str | None,
    field_type: str,
    direct: bool,
    no_repair: bool,
    no_remesh: bool,
) -> None:
    """Visualize a density field and its extraction result."""
    params = _build_params(ctx, threshold, sigma, field_type, direct, no_repair, no_remesh)
    shape = _parse_shape(shape_str) if shape_str else None

    try:
        state = load_density(input_path, field_name=field_name, shape=shape, params=params)
    except (ValueError, KeyError, ImportError) as e:
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
