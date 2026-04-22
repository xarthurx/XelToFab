"""Tests for the `xtf extrude` CLI subcommand."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh
from click.testing import CliRunner

from xeltofab.cli import main


def test_cli_extrude_beam(tmp_path: Path):
    """Full CLI round-trip: load .npy → extrude → STL."""
    field = np.zeros((10, 10), dtype=float)
    field[2:8, 2:8] = 1.0
    input_path = tmp_path / "field.npy"
    np.save(input_path, field)
    output_path = tmp_path / "part.stl"

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["extrude", str(input_path), "-o", str(output_path), "-t", "5"],
    )
    assert result.exit_code == 0, result.output
    assert output_path.exists()
    mesh = trimesh.load(output_path, force="mesh")
    assert mesh.volume > 0


def test_cli_extrude_obj_suffix(tmp_path: Path):
    """Output format follows the file suffix — no --format flag needed."""
    field = np.ones((5, 5), dtype=float)
    input_path = tmp_path / "field.npy"
    np.save(input_path, field)
    output_path = tmp_path / "part.obj"

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["extrude", str(input_path), "-o", str(output_path), "-t", "2"],
    )
    assert result.exit_code == 0, result.output
    assert output_path.exists()


def test_cli_extrude_rejects_3d_field(tmp_path: Path):
    """3D input produces a friendly error pointing to the `process` subcommand."""
    field = np.ones((5, 5, 5), dtype=float)
    input_path = tmp_path / "volume.npy"
    np.save(input_path, field)
    output_path = tmp_path / "part.stl"

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["extrude", str(input_path), "-o", str(output_path), "-t", "5"],
    )
    assert result.exit_code != 0
    assert "2D" in result.output
    assert "process" in result.output


def test_module_invocation_extrude_beam(tmp_path: Path):
    """`python -m xeltofab.cli` supports the extrude subcommand."""
    field = np.zeros((10, 10), dtype=float)
    field[2:8, 2:8] = 1.0
    input_path = tmp_path / "field.npy"
    np.save(input_path, field)
    output_path = tmp_path / "part.stl"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "xeltofab.cli",
            "extrude",
            str(input_path),
            "-o",
            str(output_path),
            "-t",
            "5",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert output_path.exists()
    mesh = trimesh.load(output_path, force="mesh")
    assert mesh.volume > 0
