# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "xeltofab",
#     "numpy",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # XelToFab Pipeline Demo

    Interactive demo of the topology optimization post-processing pipeline.

    **Pipeline:** Density field → Preprocess (smooth + threshold + morphology)
    → Extract (marching cubes/squares) → Smooth (Taubin) → Mesh / Contours

    Use the controls below to adjust parameters and watch the pipeline react.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np

    return np, plt


@app.cell
def _():
    from pathlib import Path

    from xeltofab.io import load_density
    from xeltofab.pipeline import process
    from xeltofab.state import PipelineParams, PipelineState
    from xeltofab.viz import plot_comparison, plot_density, plot_result

    return (
        Path,
        PipelineParams,
        PipelineState,
        plot_comparison,
        plot_density,
        plot_result,
        process,
    )


@app.cell
def _(Path, np):
    # Locate examples/data relative to the notebook
    _data_dir = Path(__file__).resolve().parent.parent / "data" / "examples"

    def _load_example(name):
        return np.load(_data_dir / name)

    def make_2d_circle(res=100):
        """Synthetic 2D circle."""
        y, x = np.mgrid[-1 : 1 : complex(res), -1 : 1 : complex(res * 2)]
        return (x**2 + y**2 < 0.5**2).astype(float)

    def make_3d_sphere(res=30):
        """Synthetic 3D sphere."""
        z, y, x = np.mgrid[-1 : 1 : complex(res), -1 : 1 : complex(res), -1 : 1 : complex(res)]
        return (x**2 + y**2 + z**2 < 0.5**2).astype(float)

    # Build geometry registry: synthetic + real examples
    geometry_options = {
        "Synthetic: 2D Circle": ("synthetic", make_2d_circle),
        "Synthetic: 3D Sphere": ("synthetic", make_3d_sphere),
    }
    if _data_dir.exists():
        for f in sorted(_data_dir.glob("*.npy")):
            _label = f.stem.replace("_", " ").title()
            geometry_options[_label] = ("file", f)
    return (geometry_options,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Controls
    """)
    return


@app.cell
def _(geometry_options, mo):
    _option_keys = list(geometry_options.keys())
    dim_picker = mo.ui.dropdown(
        options=_option_keys,
        value=_option_keys[0],
        label="Geometry",
    )
    return (dim_picker,)


@app.cell
def _(mo):
    threshold_slider = mo.ui.slider(start=0.05, stop=0.95, step=0.05, value=0.5, label="Threshold")
    sigma_slider = mo.ui.slider(start=0.0, stop=5.0, step=0.25, value=1.0, label="Gaussian sigma")
    taubin_slider = mo.ui.slider(start=0, stop=50, step=5, value=20, label="Taubin iterations")
    return sigma_slider, taubin_slider, threshold_slider


@app.cell(hide_code=True)
def _(dim_picker, mo, sigma_slider, taubin_slider, threshold_slider):
    mo.hstack(
        [dim_picker, threshold_slider, sigma_slider, taubin_slider],
        justify="start",
        gap=1.5,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pipeline Execution
    """)
    return


@app.cell
def _(
    PipelineParams,
    PipelineState,
    dim_picker,
    geometry_options,
    np,
    process,
    sigma_slider,
    taubin_slider,
    threshold_slider,
):
    # Load density field based on selection
    _kind, _source = geometry_options[dim_picker.value]
    density = _source() if _kind == "synthetic" else np.load(_source)

    # Build params from sliders
    params = PipelineParams(
        threshold=threshold_slider.value,
        smooth_sigma=sigma_slider.value,
        taubin_iterations=taubin_slider.value,
    )

    # Run pipeline
    state = PipelineState(density=density, params=params)
    result = process(state)
    return (result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Density Field
    """)
    return


@app.cell
def _(mo, plot_density, plt, result):
    _fig = plot_density(result)
    out = mo.as_html(_fig)
    plt.close(_fig)
    out


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extraction Result
    """)
    return


@app.cell
def _(mo, plot_result, plt, result):
    _fig = plot_result(result)
    out2 = mo.as_html(_fig)
    plt.close(_fig)
    out2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Side-by-Side Comparison
    """)
    return


@app.cell
def _(mo, plot_comparison, plt, result):
    _fig = plot_comparison(result)
    out3 = mo.as_html(_fig)
    plt.close(_fig)
    out3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pipeline Statistics
    """)
    return


@app.cell(hide_code=True)
def _(mo, result):
    _stats = []
    _stats.append(f"**Dimensions:** {result.ndim}D")
    _stats.append(f"**Density shape:** `{result.density.shape}`")
    _stats.append(
        f"**Volume fraction:** {result.volume_fraction:.4f}"
        if result.volume_fraction is not None
        else "**Volume fraction:** N/A"
    )

    if result.binary is not None:
        _fill = result.binary.sum() / result.binary.size
        _stats.append(f"**Binary fill ratio:** {_fill:.4f}")

    if result.contours is not None:
        _stats.append(f"**Contours extracted:** {len(result.contours)}")
        _total_pts = sum(c.shape[0] for c in result.contours)
        _stats.append(f"**Total contour points:** {_total_pts}")

    if result.vertices is not None:
        _stats.append(f"**Vertices:** {result.vertices.shape[0]:,}")
    if result.faces is not None:
        _stats.append(f"**Faces:** {result.faces.shape[0]:,}")
    if result.smoothed_vertices is not None:
        _stats.append("**Taubin smoothing:** applied")

    _stats.append("")
    _stats.append(
        f"**Parameters:** threshold={result.params.threshold}, "
        f"sigma={result.params.smooth_sigma}, "
        f"taubin_iter={result.params.taubin_iterations}"
    )

    mo.md("\n\n".join(_stats))
    return


if __name__ == "__main__":
    app.run()
