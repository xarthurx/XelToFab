"""Visualization functions for density fields and meshes."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from xeltofab.state import PipelineState


def plot_density(state: PipelineState) -> Figure:
    """Plot the raw density field. 2D: heatmap. 3D: mid-plane slices."""
    if state.ndim == 2:
        fig, ax = plt.subplots()
        im = ax.imshow(state.density, cmap="viridis", origin="lower", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, label="Density")
        ax.set_title("Density Field")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        d = state.density
        slices = [
            (d[d.shape[0] // 2, :, :], "XY (mid-Z)"),
            (d[:, d.shape[1] // 2, :], "XZ (mid-Y)"),
            (d[:, :, d.shape[2] // 2], "YZ (mid-X)"),
        ]
        for ax, (sl, title) in zip(axes, slices, strict=True):
            im = ax.imshow(sl, cmap="viridis", origin="lower", vmin=0, vmax=1)
            ax.set_title(title)
            fig.colorbar(im, ax=ax)
        fig.suptitle("Density Field (mid-plane slices)")
    fig.tight_layout()
    return fig


def plot_result(state: PipelineState) -> Figure:
    """Plot extraction result. 2D: contours on binary. 3D: trisurf wireframe."""
    if state.ndim == 2:
        fig, ax = plt.subplots()
        ax.imshow(state.binary, cmap="gray", origin="lower")
        if state.contours is not None:
            for contour in state.contours:
                ax.plot(contour[:, 1], contour[:, 0], "r-", linewidth=1.5)
        ax.set_title("Extracted Contours")
    else:
        vertices = state.smoothed_vertices if state.smoothed_vertices is not None else state.vertices
        if vertices is None or state.faces is None:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No mesh data", ha="center", va="center")
            return fig
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_trisurf(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            triangles=state.faces,
            alpha=0.7,
            edgecolor="k",
            linewidth=0.1,
        )
        ax.set_title("Extracted Mesh")
    fig.tight_layout()
    return fig


def plot_comparison(state: PipelineState) -> Figure:
    """Side-by-side: density field vs extraction result."""
    if state.ndim == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.imshow(state.density, cmap="viridis", origin="lower", vmin=0, vmax=1)
        ax1.set_title("Density Field")
        ax2.imshow(state.binary, cmap="gray", origin="lower")
        if state.contours is not None:
            for contour in state.contours:
                ax2.plot(contour[:, 1], contour[:, 0], "r-", linewidth=1.5)
        ax2.set_title("Extracted Contours")
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        d = state.density
        ax1.imshow(d[d.shape[0] // 2, :, :], cmap="viridis", origin="lower", vmin=0, vmax=1)
        ax1.set_title("Density (mid-Z slice)")
        vertices = state.smoothed_vertices if state.smoothed_vertices is not None else state.vertices
        if vertices is not None and state.faces is not None:
            ax2.remove()
            ax2 = fig.add_subplot(122, projection="3d")
            ax2.plot_trisurf(
                vertices[:, 0],
                vertices[:, 1],
                vertices[:, 2],
                triangles=state.faces,
                alpha=0.7,
                edgecolor="k",
                linewidth=0.1,
            )
        ax2.set_title("Extracted Mesh")
    fig.suptitle(f"Volume fraction: {state.volume_fraction:.3f}" if state.volume_fraction else "")
    fig.tight_layout()
    return fig
