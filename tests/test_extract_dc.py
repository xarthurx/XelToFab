"""Tests for Dual Contouring, Surface Nets, and manifold extraction methods."""

import unittest.mock

import numpy as np
import pytest

from xeltofab.extract import extract
from xeltofab.state import PipelineParams, PipelineState


@pytest.fixture
def sphere_sdf_30() -> np.ndarray:
    """3D SDF for a sphere: negative inside, positive outside."""
    z, y, x = np.mgrid[-1:1:30j, -1:1:30j, -1:1:30j]
    return np.sqrt(x**2 + y**2 + z**2) - 0.5


@pytest.fixture
def cube_sdf_30() -> np.ndarray:
    """3D SDF for a cube: has sharp edges that DC should preserve better than MC."""
    z, y, x = np.mgrid[-1:1:30j, -1:1:30j, -1:1:30j]
    return np.maximum(np.maximum(np.abs(x), np.abs(y)), np.abs(z)) - 0.5


class TestDualContouring:
    """Dual Contouring extraction tests."""

    def test_dc_produces_mesh(self, sphere_sdf_30: np.ndarray):
        params = PipelineParams(field_type="sdf", extraction_method="dc")
        state = PipelineState(field=sphere_sdf_30, params=params)
        result = extract(state)
        assert result.vertices is not None
        assert result.faces is not None
        assert result.vertices.shape[0] > 0
        assert result.faces.shape[0] > 0
        assert result.vertices.shape[1] == 3
        assert result.faces.shape[1] == 3

    def test_dc_vertex_count_reasonable(self, sphere_sdf_30: np.ndarray):
        """DC should produce a reasonable number of vertices (not empty, not exploded)."""
        params_dc = PipelineParams(field_type="sdf", extraction_method="dc")
        state_dc = PipelineState(field=sphere_sdf_30, params=params_dc)
        result_dc = extract(state_dc)
        # DC produces one vertex per active voxel — should be a reasonable count
        assert 100 < result_dc.vertices.shape[0] < 10000

    def test_dc_preserves_sharp_features_vs_mc(self, cube_sdf_30: np.ndarray):
        """DC should produce vertices closer to the true cube corners than MC."""
        params_dc = PipelineParams(field_type="sdf", extraction_method="dc")
        state_dc = PipelineState(field=cube_sdf_30, params=params_dc)
        result_dc = extract(state_dc)

        params_mc = PipelineParams(field_type="sdf", extraction_method="mc")
        state_mc = PipelineState(field=cube_sdf_30, params=params_mc)
        result_mc = extract(state_mc)

        dc_max = result_dc.vertices.max(axis=0).max()
        mc_max = result_mc.vertices.max(axis=0).max()
        assert dc_max > mc_max, "DC should reach closer to cube corners than MC"

    def test_dc_on_density_field(self, sphere_sdf_30: np.ndarray):
        """DC works on density fields too (though not optimal)."""
        density = (sphere_sdf_30 < 0).astype(float)
        params = PipelineParams(field_type="density", extraction_method="dc", direct_extraction=True)
        state = PipelineState(field=density, params=params)
        result = extract(state)
        assert result.vertices is not None
        assert result.vertices.shape[0] > 0


class TestSurfaceNets:
    """Naive Surface Nets extraction tests."""

    def test_surfnets_produces_mesh(self, sphere_sdf_30: np.ndarray):
        params = PipelineParams(field_type="sdf", extraction_method="surfnets")
        state = PipelineState(field=sphere_sdf_30, params=params)
        result = extract(state)
        assert result.vertices is not None
        assert result.faces is not None
        assert result.vertices.shape[0] > 0
        assert result.faces.shape[0] > 0
        assert result.vertices.shape[1] == 3
        assert result.faces.shape[1] == 3

    def test_surfnets_produces_reasonable_sphere(self, sphere_sdf_30: np.ndarray):
        """Surface nets vertices should approximate the sphere reasonably well."""
        params_sn = PipelineParams(field_type="sdf", extraction_method="surfnets")
        state_sn = PipelineState(field=sphere_sdf_30, params=params_sn)
        result_sn = extract(state_sn)

        center = np.array([14.5, 14.5, 14.5])
        radii = np.linalg.norm(result_sn.vertices - center, axis=1)
        # Vertices should roughly approximate a sphere (expected radius ~7.25 grid units)
        assert radii.mean() > 5.0
        assert radii.mean() < 10.0
        assert np.std(radii) < 1.0  # should be roughly spherical


class TestManifoldExtraction:
    """manifold3d extraction tests (requires manifold3d optional dep)."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_manifold3d(self):
        pytest.importorskip("manifold3d")

    def test_manifold_produces_mesh(self, sphere_sdf_30: np.ndarray):
        params = PipelineParams(field_type="sdf", extraction_method="manifold")
        state = PipelineState(field=sphere_sdf_30, params=params)
        result = extract(state)
        assert result.vertices is not None
        assert result.faces is not None
        assert result.vertices.shape[0] > 0
        assert result.faces.shape[0] > 0
        assert result.vertices.shape[1] == 3
        assert result.faces.shape[1] == 3

    def test_manifold_on_density_field(self):
        """manifold extraction works on density fields at level=0.5."""
        z, y, x = np.mgrid[-1:1:20j, -1:1:20j, -1:1:20j]
        field = (x**2 + y**2 + z**2 < 0.5**2).astype(float)
        params = PipelineParams(field_type="density", extraction_method="manifold", direct_extraction=True)
        state = PipelineState(field=field, params=params)
        result = extract(state)
        assert result.vertices is not None
        assert result.vertices.shape[0] > 0

    def test_manifold_watertight(self, sphere_sdf_30: np.ndarray):
        """manifold3d output should be watertight."""
        import trimesh

        params = PipelineParams(field_type="sdf", extraction_method="manifold")
        state = PipelineState(field=sphere_sdf_30, params=params)
        result = extract(state)
        mesh = trimesh.Trimesh(vertices=result.vertices, faces=result.faces, process=False)
        assert mesh.is_watertight


def test_manifold_import_error():
    """Clear error message when manifold3d is not installed."""
    with unittest.mock.patch.dict("sys.modules", {"manifold3d": None}):
        z, y, x = np.mgrid[-1:1:10j, -1:1:10j, -1:1:10j]
        sdf = np.sqrt(x**2 + y**2 + z**2) - 0.5
        params = PipelineParams(field_type="sdf", extraction_method="manifold")
        state = PipelineState(field=sdf, params=params)
        with pytest.raises(ImportError, match="manifold3d not installed"):
            extract(state)


class TestEmptyMeshGuard:
    """Tests for the empty mesh post-extraction guard."""

    def test_empty_mesh_raises(self):
        """All-positive field (no sign changes) should raise ValueError."""
        field = np.ones((10, 10, 10))  # all positive, no zero crossing
        params = PipelineParams(field_type="sdf", extraction_method="dc")
        state = PipelineState(field=field, params=params)
        with pytest.raises(ValueError, match="Extraction produced no geometry"):
            extract(state)

    def test_empty_mesh_surfnets(self):
        """Surface nets on all-positive field should also raise."""
        field = np.ones((10, 10, 10))
        params = PipelineParams(field_type="sdf", extraction_method="surfnets")
        state = PipelineState(field=field, params=params)
        with pytest.raises(ValueError, match="Extraction produced no geometry"):
            extract(state)


class TestGPUFallback:
    """Tests for GPU DC fallback path."""

    def test_gpu_fallback_when_no_cuda(self, sphere_sdf_30: np.ndarray):
        """When torch.cuda.is_available() returns False, falls back to CPU DC."""
        with unittest.mock.patch.dict("sys.modules", {"torch": unittest.mock.MagicMock()}):
            import sys
            mock_torch = sys.modules["torch"]
            mock_torch.cuda.is_available.return_value = False
            # The fallback should use CPU DC and produce valid output
            params = PipelineParams(field_type="sdf", extraction_method="dc")
            state = PipelineState(field=sphere_sdf_30, params=params)
            result = extract(state)
            assert result.vertices is not None
            assert result.vertices.shape[0] > 0

    def test_gpu_fallback_when_no_torch(self, sphere_sdf_30: np.ndarray):
        """When torch is not installed, falls back to CPU DC."""
        with unittest.mock.patch.dict("sys.modules", {"torch": None}):
            params = PipelineParams(field_type="sdf", extraction_method="dc")
            state = PipelineState(field=sphere_sdf_30, params=params)
            result = extract(state)
            assert result.vertices is not None
            assert result.vertices.shape[0] > 0


class TestExtractionMethodDispatch:
    """Verify correct dispatch to extraction backends."""

    def test_mc_is_default_for_density(self):
        """Density fields use MC by default."""
        params = PipelineParams(field_type="density")
        assert params.extraction_method == "mc"

    def test_dc_is_default_for_sdf(self):
        """SDF fields use DC by default."""
        params = PipelineParams(field_type="sdf")
        assert params.extraction_method == "dc"

    def test_2d_always_uses_marching_squares(self):
        """2D fields always use marching squares regardless of extraction_method."""
        y, x = np.mgrid[-1:1:50j, -1:1:50j]
        sdf_2d = np.sqrt(x**2 + y**2) - 0.5
        params = PipelineParams(field_type="sdf", extraction_method="dc")
        state = PipelineState(field=sdf_2d, params=params)
        result = extract(state)
        assert result.contours is not None
        assert result.vertices is None


class TestEndToEnd:
    """End-to-end pipeline tests with different extraction methods."""

    def test_process_with_dc(self, sphere_sdf_30: np.ndarray):
        """Full pipeline with DC extraction produces valid output."""
        from xeltofab.pipeline import process

        params = PipelineParams(field_type="sdf", extraction_method="dc")
        state = PipelineState(field=sphere_sdf_30, params=params)
        result = process(state)
        assert result.vertices is not None
        assert result.faces is not None
        assert result.vertices.shape[0] > 0

    def test_process_with_surfnets(self, sphere_sdf_30: np.ndarray):
        """Full pipeline with Surface Nets extraction produces valid output."""
        from xeltofab.pipeline import process

        params = PipelineParams(field_type="sdf", extraction_method="surfnets")
        state = PipelineState(field=sphere_sdf_30, params=params)
        result = process(state)
        assert result.vertices is not None
        assert result.faces is not None

    def test_process_dc_no_repair(self, sphere_sdf_30: np.ndarray):
        """DC extraction with repair disabled still produces output."""
        from xeltofab.pipeline import process

        params = PipelineParams(field_type="sdf", extraction_method="dc", repair=False)
        state = PipelineState(field=sphere_sdf_30, params=params)
        result = process(state)
        assert result.vertices is not None


@pytest.mark.parametrize("method", ["mc", "dc"])
def test_extract_3d_direct_sdf_parametrized(method: str):
    """Direct extraction from continuous SDF at level=0 — both MC and DC."""
    z, y, x = np.mgrid[-1:1:30j, -1:1:30j, -1:1:30j]
    sphere_sdf = np.sqrt(x**2 + y**2 + z**2) - 0.5
    params = PipelineParams(field_type="sdf", extraction_method=method)
    state = PipelineState(field=sphere_sdf, params=params)
    result = extract(state)
    assert result.vertices is not None
    assert result.faces is not None
    assert result.vertices.shape[0] > 0
    assert result.faces.shape[0] > 0


@pytest.mark.parametrize("method", ["mc", "dc"])
def test_extract_3d_direct_custom_level_parametrized(method: str):
    """Direct extraction at a custom level — both MC and DC."""
    z, y, x = np.mgrid[-1:1:30j, -1:1:30j, -1:1:30j]
    sphere_sdf = np.sqrt(x**2 + y**2 + z**2) - 0.5

    params_offset = PipelineParams(field_type="sdf", extraction_level=0.1, extraction_method=method)
    state_offset = PipelineState(field=sphere_sdf, params=params_offset)
    result_offset = extract(state_offset)
    assert result_offset.vertices is not None

    params_zero = PipelineParams(field_type="sdf", extraction_level=0.0, extraction_method=method)
    state_zero = PipelineState(field=sphere_sdf, params=params_zero)
    result_zero = extract(state_zero)

    center = np.array([14.5, 14.5, 14.5])
    mean_r_zero = float(np.mean(np.linalg.norm(result_zero.vertices - center, axis=1)))
    mean_r_offset = float(np.mean(np.linalg.norm(result_offset.vertices - center, axis=1)))
    assert mean_r_offset > mean_r_zero
