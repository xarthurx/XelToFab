# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "xeltocad",
#     "numpy",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # XelToCAD Pipeline Demo

        Interactive demo of the topology optimization post-processing pipeline.

        **Pipeline:** Density field → Preprocess (smooth + threshold + morphology)
        → Extract (marching cubes/squares) → Smooth (Taubin) → Mesh / Contours

        Use the controls below to adjust parameters and watch the pipeline react.
        """
    )
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

    return matplotlib, np, plt


@app.cell
def _():
    from xeltocad.pipeline import process
    from xeltocad.state import PipelineParams, PipelineState
    from xeltocad.viz import plot_comparison, plot_density, plot_result

    return PipelineParams, PipelineState, plot_comparison, plot_density, plot_result, process


@app.cell
def _(np):
    def make_2d_circle(res=100):
        """2D density field: filled circle."""
        y, x = np.mgrid[-1 : 1 : complex(res), -1 : 1 : complex(res * 2)]
        return (x**2 + y**2 < 0.5**2).astype(float)

    def make_3d_sphere(res=30):
        """3D density field: filled sphere."""
        z, y, x = np.mgrid[-1 : 1 : complex(res), -1 : 1 : complex(res), -1 : 1 : complex(res)]
        return (x**2 + y**2 + z**2 < 0.5**2).astype(float)

    def make_2d_bridge(res=100):
        """2D density field: bridge-like structure with supports and deck."""
        field = np.zeros((res // 2, res))
        # deck
        field[res // 2 - 8 : res // 2 - 2, 5:-5] = 1.0
        # left support
        field[5 : res // 2 - 2, 5:15] = 1.0
        # right support
        field[5 : res // 2 - 2, -15:-5] = 1.0
        # middle support
        field[10 : res // 2 - 2, res // 2 - 5 : res // 2 + 5] = 0.8
        # add some noise for realism
        field += np.random.rand(*field.shape) * 0.15
        return np.clip(field, 0, 1)

    return make_2d_bridge, make_2d_circle, make_3d_sphere


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## Controls")
    return


@app.cell
def _(mo):
    dim_picker = mo.ui.dropdown(
        options={"2D Circle": "circle", "2D Bridge": "bridge", "3D Sphere": "sphere"},
        value="circle",
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
    mo.md(r"## Pipeline Execution")
    return


@app.cell
def _(
    PipelineParams,
    PipelineState,
    dim_picker,
    make_2d_bridge,
    make_2d_circle,
    make_3d_sphere,
    process,
    sigma_slider,
    taubin_slider,
    threshold_slider,
):
    # Generate density field based on selection
    _generators = {
        "circle": make_2d_circle,
        "bridge": make_2d_bridge,
        "sphere": make_3d_sphere,
    }
    density = _generators[dim_picker.value]()

    # Build params from sliders
    params = PipelineParams(
        threshold=threshold_slider.value,
        smooth_sigma=sigma_slider.value,
        taubin_iterations=taubin_slider.value,
    )

    # Run pipeline
    state = PipelineState(density=density, params=params)
    result = process(state)
    return density, params, result, state


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## Density Field")
    return


@app.cell
def _(mo, plt, plot_density, result):
    _fig = plot_density(result)
    _out = mo.as_html(_fig)
    plt.close(_fig)
    return (_out,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## Extraction Result")
    return


@app.cell
def _(mo, plt, plot_result, result):
    _fig = plot_result(result)
    _out2 = mo.as_html(_fig)
    plt.close(_fig)
    return (_out2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## Side-by-Side Comparison")
    return


@app.cell
def _(mo, plt, plot_comparison, result):
    _fig = plot_comparison(result)
    _out3 = mo.as_html(_fig)
    plt.close(_fig)
    return (_out3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## Pipeline Statistics")
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
