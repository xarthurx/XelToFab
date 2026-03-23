"""CLI entrypoint for XelToFab."""

from __future__ import annotations

from pathlib import Path

import click

from xeltofab.field_plots import plot_comparison
from xeltofab.io import load_field, save_mesh
from xeltofab.loaders import get_supported_formats
from xeltofab.pipeline import process
from xeltofab.state import PipelineParams


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
    no_decimate: bool,
    smoothing: str,
    extraction_method: str,
) -> PipelineParams:
    """Build PipelineParams, only passing values explicitly set by the user.

    This preserves PipelineParams smart defaults (e.g., SDF auto-enables
    direct extraction and disables Gaussian smoothing, SDF auto-selects DC).
    """
    kwargs: dict = {"field_type": field_type}
    source = click.core.ParameterSource.COMMANDLINE
    # Only pass values explicitly set on CLI — preserve PipelineParams smart defaults
    if ctx.get_parameter_source("threshold") == source:
        kwargs["threshold"] = threshold
    if ctx.get_parameter_source("sigma") == source:
        kwargs["smooth_sigma"] = sigma
    if ctx.get_parameter_source("direct") == source:
        kwargs["direct_extraction"] = direct
    if ctx.get_parameter_source("extraction_method") == source:
        kwargs["extraction_method"] = extraction_method
    if ctx.get_parameter_source("smoothing") == source:
        kwargs["smoothing_method"] = smoothing
    if no_repair:
        kwargs["repair"] = False
    if no_remesh:
        kwargs["remesh"] = False
    if no_decimate:
        kwargs["decimate"] = False
    return PipelineParams(**kwargs)


@click.group()
def main() -> None:
    """XelToFab — Design fields to fabrication-ready geometry."""


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", "output_path", type=click.Path(path_type=Path), required=True)
@click.option("--threshold", type=float, default=0.5, help="Field threshold [0-1]")
@click.option("--sigma", type=float, default=1.0, help="Gaussian smoothing sigma")
@click.option("-f", "--field-name", default=None, help="Field/variable name to extract from input file")
@click.option("--shape", "shape_str", default=None, help="Grid shape for flat data, e.g. 100x200 or 10x20x30")
@click.option("--field-type", type=click.Choice(["density", "sdf"]), default="density", help="Input field type")
@click.option("--direct", is_flag=True, help="Direct extraction from continuous field (skip preprocessing)")
@click.option("--no-repair", is_flag=True, help="Disable watertight mesh repair")
@click.option("--no-remesh", is_flag=True, help="Disable isotropic remeshing")
@click.option("--no-decimate", is_flag=True, help="Disable QEM mesh decimation")
@click.option("--smoothing", type=click.Choice(["taubin", "bilateral"]), default="taubin", help="Mesh smoothing method")
@click.option(
    "--extraction-method",
    type=click.Choice(["mc", "dc", "surfnets", "manifold"]),
    default="mc",
    help="Extraction: mc=marching cubes, dc=dual contouring, surfnets=surface nets, manifold=manifold3d",
)
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
    no_decimate: bool,
    smoothing: str,
    extraction_method: str,
    viz: bool,
) -> None:
    """Process a scalar field into a mesh."""
    params = _build_params(
        ctx, threshold, sigma, field_type, direct, no_repair, no_remesh, no_decimate, smoothing, extraction_method
    )
    shape = _parse_shape(shape_str) if shape_str else None

    try:
        state = load_field(input_path, field_name=field_name, shape=shape, params=params)
    except (ValueError, KeyError, ImportError) as e:
        raise click.ClickException(str(e)) from None

    state = process(state)
    try:
        save_mesh(state, output_path)
    except ValueError as e:
        raise click.ClickException(str(e)) from None
    click.echo(f"Saved mesh to {output_path}")

    if viz:
        import matplotlib.pyplot as plt

        fig = plot_comparison(state)
        viz_path = output_path.with_suffix(".png")
        fig.savefig(viz_path, dpi=150)
        plt.close(fig)
        click.echo(f"Saved visualization to {viz_path}")


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", "output_path", type=click.Path(path_type=Path), default=None)
@click.option("--threshold", type=float, default=0.5, help="Field threshold [0-1]")
@click.option("--sigma", type=float, default=1.0, help="Gaussian smoothing sigma")
@click.option("-f", "--field-name", default=None, help="Field/variable name to extract from input file")
@click.option("--shape", "shape_str", default=None, help="Grid shape for flat data, e.g. 100x200 or 10x20x30")
@click.option("--field-type", type=click.Choice(["density", "sdf"]), default="density", help="Input field type")
@click.option("--direct", is_flag=True, help="Direct extraction from continuous field (skip preprocessing)")
@click.option("--no-repair", is_flag=True, help="Disable watertight mesh repair")
@click.option("--no-remesh", is_flag=True, help="Disable isotropic remeshing")
@click.option("--no-decimate", is_flag=True, help="Disable QEM mesh decimation")
@click.option("--smoothing", type=click.Choice(["taubin", "bilateral"]), default="taubin", help="Mesh smoothing method")
@click.option(
    "--extraction-method",
    type=click.Choice(["mc", "dc", "surfnets", "manifold"]),
    default="mc",
    help="Extraction: mc=marching cubes, dc=dual contouring, surfnets=surface nets, manifold=manifold3d",
)
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
    no_decimate: bool,
    smoothing: str,
    extraction_method: str,
) -> None:
    """Visualize a scalar field and its extraction result."""
    params = _build_params(
        ctx, threshold, sigma, field_type, direct, no_repair, no_remesh, no_decimate, smoothing, extraction_method
    )
    shape = _parse_shape(shape_str) if shape_str else None

    try:
        state = load_field(input_path, field_name=field_name, shape=shape, params=params)
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
