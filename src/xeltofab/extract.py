"""Mesh/contour extraction from preprocessed binary fields."""

from __future__ import annotations

from skimage.measure import find_contours, marching_cubes

from xeltofab.state import PipelineState


def extract(state: PipelineState) -> PipelineState:
    """Extract mesh (3D) or contours (2D) from the binary field."""
    if state.binary is None:
        raise ValueError("binary field is None — run preprocess() first")

    if state.ndim == 2:
        return _extract_2d(state)
    return _extract_3d(state)


def _extract_2d(state: PipelineState) -> PipelineState:
    """Extract contours from 2D binary field using marching squares."""
    contours = find_contours(state.binary.astype(float), level=0.5)
    return state.model_copy(update={"contours": contours})


def _extract_3d(state: PipelineState) -> PipelineState:
    """Extract triangle mesh from 3D binary field using marching cubes."""
    vertices, faces, _, _ = marching_cubes(state.binary.astype(float), level=0.5)
    return state.model_copy(update={"vertices": vertices, "faces": faces})
