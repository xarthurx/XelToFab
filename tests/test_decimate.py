# tests/test_decimate.py
import numpy as np
import pytest
import trimesh

pytest.importorskip("pyfqmr")

from xeltofab.decimate import decimate
from xeltofab.extract import extract
from xeltofab.preprocess import preprocess
from xeltofab.state import PipelineParams, PipelineState


def _make_state(sphere_field: np.ndarray, **param_overrides) -> PipelineState:
    params = PipelineParams(**param_overrides)
    return extract(preprocess(PipelineState(field=sphere_field, params=params)))


def test_decimate_reduces_face_count(sphere_field: np.ndarray):
    state = _make_state(sphere_field)
    result = decimate(state)
    assert result.faces is not None
    assert len(result.faces) < len(state.faces)
    assert result.smoothed_vertices is None


def test_decimate_ratio(sphere_field: np.ndarray):
    state = _make_state(sphere_field, decimate_ratio=0.5)
    original_count = len(state.faces)
    result = decimate(state)
    # pyfqmr may not hit exact target, allow 20% tolerance
    assert len(result.faces) < original_count * 0.7


def test_decimate_target_faces(sphere_field: np.ndarray):
    state = _make_state(sphere_field, target_faces=100)
    result = decimate(state)
    # pyfqmr may undershoot slightly; check within range
    assert len(result.faces) <= 120


def test_decimate_preserves_volume(sphere_field: np.ndarray):
    state = _make_state(sphere_field, decimate_ratio=0.5)
    original_mesh = trimesh.Trimesh(vertices=state.vertices, faces=state.faces)
    result = decimate(state)
    decimated_mesh = trimesh.Trimesh(vertices=result.vertices, faces=result.faces)
    ratio = decimated_mesh.volume / original_mesh.volume
    assert ratio > 0.9, f"Volume ratio {ratio:.3f} is too low — decimation destroyed the mesh"


def test_decimate_2d_is_noop(circle_field: np.ndarray):
    state = extract(preprocess(PipelineState(field=circle_field)))
    result = decimate(state)
    assert result.contours is not None
    assert result.vertices is None


def test_decimate_disabled(sphere_field: np.ndarray):
    state = _make_state(sphere_field, decimate=False)
    result = decimate(state)
    assert np.array_equal(result.faces, state.faces)
    assert np.array_equal(result.vertices, state.vertices)
