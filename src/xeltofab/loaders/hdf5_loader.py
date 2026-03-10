"""HDF5 and XDMF loader for scalar fields."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np

# Same auto-detect names as MATLAB loader
_KNOWN_NAMES = ("xPhys", "densities", "x", "rho", "dc", "density")


def _find_datasets(group: h5py.Group, prefix: str = "") -> list[str]:
    """Recursively collect all dataset paths in an HDF5 file."""
    paths = []
    for key in group:
        full_path = f"{prefix}/{key}" if prefix else key
        item = group[key]
        if isinstance(item, h5py.Dataset):
            paths.append(full_path)
        elif isinstance(item, h5py.Group):
            paths.extend(_find_datasets(item, full_path))
    return paths


def _load_h5(path: Path, field_name: str | None) -> np.ndarray:
    """Load from a raw HDF5 file."""
    with h5py.File(path, "r") as f:
        if field_name is not None:
            if field_name not in f:
                all_datasets = _find_datasets(f)
                raise KeyError(f"Field '{field_name}' not found in {path.name}. Available: {all_datasets}")
            return np.asarray(f[field_name], dtype=np.float64)

        all_datasets = _find_datasets(f)

        # Auto-detect by known names (check leaf name)
        for ds_path in all_datasets:
            leaf = ds_path.rsplit("/", 1)[-1]
            if leaf in _KNOWN_NAMES:
                return np.asarray(f[ds_path], dtype=np.float64)

        # Single dataset fallback
        if len(all_datasets) == 1:
            return np.asarray(f[all_datasets[0]], dtype=np.float64)

        raise ValueError(f"Multiple datasets found in {path.name}: {all_datasets}\nSpecify which one with --field-name")


def _strip_ns(tag: str) -> str:
    """Strip XML namespace prefix from a tag, e.g. '{http://...}DataItem' -> 'DataItem'."""
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def _parse_hdf_ref(text: str) -> tuple[str, str] | None:
    """Parse an XDMF HDF reference like 'file.h5:/dataset'.

    Uses rpartition to handle Windows drive-letter paths (e.g. C:/dir/file.h5:/ds).
    """
    h5_filename, sep, dataset_path = text.strip().rpartition(":")
    if not sep or not dataset_path:
        return None
    return h5_filename, dataset_path


def _load_xdmf(path: Path, field_name: str | None) -> np.ndarray:
    """Load from an XDMF file (XML metadata pointing to HDF5 data)."""
    tree = ET.parse(path)
    root = tree.getroot()

    # Only search DataItems under Attribute elements (skip Geometry, Topology, etc.)
    for elem in root.iter():
        if _strip_ns(elem.tag) != "Attribute":
            continue

        attr_name = elem.get("Name", "")
        if field_name is not None and attr_name != field_name:
            continue

        for data_item in elem.iter():
            if _strip_ns(data_item.tag) != "DataItem":
                continue
            fmt = data_item.get("Format", "")
            if fmt.upper() != "HDF":
                continue
            if data_item.text is None:
                continue

            parsed = _parse_hdf_ref(data_item.text)
            if parsed is None:
                continue
            h5_filename, dataset_path = parsed

            h5_path = path.parent / h5_filename
            with h5py.File(h5_path, "r") as f:
                dataset_path = dataset_path.lstrip("/")
                return np.asarray(f[dataset_path], dtype=np.float64)

    raise ValueError(f"No HDF Attribute data items found in {path.name}")


def load(path: Path, field_name: str | None, shape: tuple[int, ...] | None) -> np.ndarray:
    """Load scalar field from HDF5 or XDMF file."""
    if path.suffix.lower() == ".xdmf":
        return _load_xdmf(path, field_name)
    return _load_h5(path, field_name)
