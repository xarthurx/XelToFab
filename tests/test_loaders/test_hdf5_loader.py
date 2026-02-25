"""Tests for HDF5/XDMF loader."""

from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from xeltocad.loaders.hdf5_loader import load  # noqa: E402


def test_load_h5_auto_detect(tmp_path: Path):
    arr = np.random.rand(20, 40)
    path = tmp_path / "test.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("density", data=arr)
    result = load(path, field_name=None, shape=None)
    np.testing.assert_array_almost_equal(result, arr)


def test_load_h5_explicit_field(tmp_path: Path):
    arr = np.random.rand(10, 20)
    path = tmp_path / "test.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("custom", data=arr)
        f.create_dataset("other", data=np.zeros((5, 5)))
    result = load(path, field_name="custom", shape=None)
    np.testing.assert_array_almost_equal(result, arr)


def test_load_h5_nested_group(tmp_path: Path):
    """FEniCS-style nested path like /Function/0."""
    arr = np.random.rand(10, 10)
    path = tmp_path / "test.h5"
    with h5py.File(path, "w") as f:
        grp = f.create_group("Function")
        grp.create_dataset("0", data=arr)
    result = load(path, field_name="Function/0", shape=None)
    np.testing.assert_array_almost_equal(result, arr)


def test_load_h5_missing_field_raises(tmp_path: Path):
    path = tmp_path / "test.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("real_field", data=np.zeros((5, 5)))
    with pytest.raises(KeyError, match="nonexistent"):
        load(path, field_name="nonexistent", shape=None)


def test_load_h5_multiple_unknown_datasets_raises(tmp_path: Path):
    path = tmp_path / "test.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("foo", data=np.zeros((5, 5)))
        f.create_dataset("bar", data=np.ones((5, 5)))
    with pytest.raises(ValueError, match="Multiple datasets"):
        load(path, field_name=None, shape=None)


def test_load_xdmf(tmp_path: Path):
    """XDMF file pointing to an HDF5 dataset."""
    arr = np.random.rand(10, 20).astype(np.float64)
    h5_path = tmp_path / "data.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("density", data=arr)

    xdmf_path = tmp_path / "data.xdmf"
    xdmf_path.write_text("""\
<?xml version="1.0" ?>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="mesh" GridType="Uniform">
      <Topology TopologyType="2DRectMesh" Dimensions="11 21"/>
      <Attribute Name="density" Center="Cell">
        <DataItem Format="HDF" Dimensions="10 20" DataType="Float" Precision="8">
          data.h5:/density
        </DataItem>
      </Attribute>
    </Grid>
  </Domain>
</Xdmf>
""")

    result = load(xdmf_path, field_name=None, shape=None)
    np.testing.assert_array_almost_equal(result, arr)


def test_load_xdmf_geometry_before_density(tmp_path: Path):
    """XDMF with geometry DataItem before density — must return density, not coords."""
    density = np.random.rand(10, 20).astype(np.float64)
    coords = np.random.rand(11 * 21, 2).astype(np.float64)  # geometry is different shape
    h5_path = tmp_path / "data.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("geometry", data=coords)
        f.create_dataset("density", data=density)

    xdmf_path = tmp_path / "data.xdmf"
    xdmf_path.write_text("""\
<?xml version="1.0" ?>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="mesh" GridType="Uniform">
      <Topology TopologyType="2DRectMesh" Dimensions="11 21"/>
      <Geometry GeometryType="XY">
        <DataItem Format="HDF" Dimensions="231 2" DataType="Float" Precision="8">
          data.h5:/geometry
        </DataItem>
      </Geometry>
      <Attribute Name="density" Center="Cell">
        <DataItem Format="HDF" Dimensions="10 20" DataType="Float" Precision="8">
          data.h5:/density
        </DataItem>
      </Attribute>
    </Grid>
  </Domain>
</Xdmf>
""")

    result = load(xdmf_path, field_name=None, shape=None)
    np.testing.assert_array_almost_equal(result, density)


def test_load_xdmf_with_namespace(tmp_path: Path):
    """XDMF with default XML namespace should still parse."""
    arr = np.random.rand(5, 10).astype(np.float64)
    h5_path = tmp_path / "data.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("density", data=arr)

    xdmf_path = tmp_path / "data.xdmf"
    xdmf_path.write_text("""\
<?xml version="1.0" ?>
<Xdmf xmlns="http://www.xdmf.org/2.0" Version="3.0">
  <Domain>
    <Grid Name="mesh" GridType="Uniform">
      <Topology TopologyType="2DRectMesh" Dimensions="6 11"/>
      <Attribute Name="density" Center="Cell">
        <DataItem Format="HDF" Dimensions="5 10" DataType="Float" Precision="8">
          data.h5:/density
        </DataItem>
      </Attribute>
    </Grid>
  </Domain>
</Xdmf>
""")

    result = load(xdmf_path, field_name=None, shape=None)
    np.testing.assert_array_almost_equal(result, arr)


def test_load_h5_returns_float64(tmp_path: Path):
    arr = np.random.rand(8, 8).astype(np.float32)
    path = tmp_path / "test.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("density", data=arr)
    result = load(path, field_name=None, shape=None)
    assert result.dtype == np.float64
