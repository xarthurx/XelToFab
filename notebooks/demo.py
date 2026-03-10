# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "xeltofab",
#     "numpy",
#     "matplotlib",
#     "plotly",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # XelToFab Pipeline Demo

    Interactive demo of the design field post-processing pipeline.

    **Pipeline:** Scalar field → Preprocess (smooth + threshold + morphology)
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

    from xeltofab.io import load_field
    from xeltofab.pipeline import process
    from xeltofab.state import PipelineParams, PipelineState
    from xeltofab.field_plots import plot_comparison, plot_field, plot_result

    return (
        Path,
        PipelineParams,
        PipelineState,
        plot_comparison,
        plot_field,
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
        """Synthetic 2D circle with smooth boundary (mimics TO density field)."""
        y, x = np.mgrid[-1 : 1 : complex(res), -1 : 1 : complex(res)]
        r = np.sqrt(x**2 + y**2)
        # Smooth falloff across boundary using a sigmoid-like transition
        width = 0.08  # transition width
        field = 0.5 * (1.0 - np.tanh((r - 0.5) / width))
        return np.clip(field, 0.0, 1.0)

    def make_3d_sphere(res=60):
        """Synthetic 3D sphere with smooth boundary (mimics TO density field)."""
        z, y, x = np.mgrid[-1 : 1 : complex(res), -1 : 1 : complex(res), -1 : 1 : complex(res)]
        r = np.sqrt(x**2 + y**2 + z**2)
        width = 0.08  # transition width matching 2D circle
        field = 0.5 * (1.0 - np.tanh((r - 0.5) / width))
        return np.clip(field, 0.0, 1.0)

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

    **Thr** — field threshold for binarisation &nbsp;|&nbsp;
    **σ** — Gaussian smoothing radius &nbsp;|&nbsp;
    **Tau** — Taubin smoothing iter. on the extracted mesh
    """)
    return


@app.cell
def _(geometry_options, mo):
    _option_keys = list(geometry_options.keys())
    dim_picker = mo.ui.dropdown(
        options=_option_keys,
        value=_option_keys[0],
        label="Geo",
    )
    return (dim_picker,)


@app.cell
def _(mo):
    threshold_slider = mo.ui.slider(start=0.05, stop=0.95, step=0.05, value=0.5, label="Thr")
    sigma_slider = mo.ui.slider(start=0.0, stop=5.0, step=0.25, value=1.0, label="σ")
    taubin_slider = mo.ui.slider(start=0, stop=50, step=5, value=20, label="Tau")
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
    # Load field based on selection
    _kind, _source = geometry_options[dim_picker.value]
    field = _source() if _kind == "synthetic" else np.load(_source)

    # Build params from sliders
    params = PipelineParams(
        threshold=threshold_slider.value,
        smooth_sigma=sigma_slider.value,
        taubin_iterations=taubin_slider.value,
    )

    # Run pipeline
    state = PipelineState(field=field, params=params)
    result = process(state)
    return (result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Input Field
    """)
    return


@app.cell
def _(mo, plot_field, plt, result):
    _fig = plot_field(result)
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
    if result.ndim == 3 and result.vertices is not None and result.faces is not None:
        import plotly.graph_objects as _go

        _verts = result.smoothed_vertices if result.smoothed_vertices is not None else result.vertices
        _faces = result.faces
        _fig3d = _go.Figure(data=[_go.Mesh3d(
            x=_verts[:, 0], y=_verts[:, 1], z=_verts[:, 2],
            i=_faces[:, 0], j=_faces[:, 1], k=_faces[:, 2],
            color="steelblue", opacity=0.9,
            lighting=dict(ambient=0.4, diffuse=0.6, specular=0.3),
        )])
        _fig3d.update_layout(
            title="Extracted Mesh (interactive)",
            scene=dict(
                aspectmode="data",
                camera=dict(projection=dict(type="orthographic")),
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            height=500,
        )
        out2 = mo.ui.plotly(_fig3d)
    else:
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
    if result.ndim == 3 and result.vertices is not None and result.faces is not None:
        import plotly.graph_objects as _go

        _verts = result.smoothed_vertices if result.smoothed_vertices is not None else result.vertices
        _faces = result.faces
        _d = result.field
        _mid_slice = _d[_d.shape[0] // 2, :, :]

        _panel_h = 350  # consistent height for all three panels

        # Density mid-slice (matplotlib)
        _fig_den, _ax_den = plt.subplots(figsize=(4, 4))
        _ax_den.imshow(_mid_slice, cmap="viridis", origin="lower", vmin=0, vmax=1, aspect="equal")
        _ax_den.set_title("Field (mid-Z)")
        _fig_den.tight_layout()
        _density_html = mo.as_html(_fig_den)
        plt.close(_fig_den)

        # Binary mid-slice (matplotlib)
        _bin_slice = result.binary[result.binary.shape[0] // 2, :, :]
        _fig_bin, _ax_bin = plt.subplots(figsize=(4, 4))
        _ax_bin.imshow(_bin_slice, cmap="gray", origin="lower", vmin=0, vmax=1, aspect="equal")
        _ax_bin.set_title("Binary (mid-Z)")
        _fig_bin.tight_layout()
        _binary_html = mo.as_html(_fig_bin)
        plt.close(_fig_bin)

        # 3D mesh (plotly)
        _fig3d = _go.Figure(data=[_go.Mesh3d(
            x=_verts[:, 0], y=_verts[:, 1], z=_verts[:, 2],
            i=_faces[:, 0], j=_faces[:, 1], k=_faces[:, 2],
            color="steelblue", opacity=0.9,
            lighting=dict(ambient=0.4, diffuse=0.6, specular=0.3),
        )])
        _fig3d.update_layout(
            title="Extracted Mesh",
            scene=dict(
                aspectmode="data",
                camera=dict(projection=dict(type="orthographic")),
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            height=_panel_h,
        )
        _mesh_html = mo.ui.plotly(_fig3d)

        out3 = mo.hstack([_density_html, _binary_html, _mesh_html], widths=[1, 1, 1])
    else:
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
    _stats.append(f"**Field shape:** `{result.field.shape}`")
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
