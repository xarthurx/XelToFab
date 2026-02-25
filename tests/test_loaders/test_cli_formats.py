"""Tests for CLI format-related features."""

from pathlib import Path

import numpy as np
import scipy.io
from click.testing import CliRunner

from xeltocad.cli import main


def test_cli_formats_subcommand():
    runner = CliRunner()
    result = runner.invoke(main, ["formats"])
    assert result.exit_code == 0
    assert "numpy" in result.output
    assert ".npy" in result.output


def test_cli_process_mat_file(tmp_path: Path, small_sphere_density: np.ndarray):
    """Process a .mat file through the CLI."""
    input_path = tmp_path / "test.mat"
    scipy.io.savemat(input_path, {"xPhys": small_sphere_density})
    output_path = tmp_path / "output.stl"

    runner = CliRunner()
    result = runner.invoke(main, ["process", str(input_path), "-o", str(output_path)])
    assert result.exit_code == 0
    assert output_path.exists()


def test_cli_process_mat_with_field_name(tmp_path: Path, small_sphere_density: np.ndarray):
    input_path = tmp_path / "test.mat"
    scipy.io.savemat(input_path, {"custom_name": small_sphere_density})
    output_path = tmp_path / "output.stl"

    runner = CliRunner()
    result = runner.invoke(main, ["process", str(input_path), "-o", str(output_path), "--field-name", "custom_name"])
    assert result.exit_code == 0
    assert output_path.exists()


def test_cli_process_csv_with_shape(tmp_path: Path):
    arr = np.random.rand(50, 100)
    input_path = tmp_path / "test.csv"
    np.savetxt(input_path, arr.ravel(), delimiter=",")
    output_path = tmp_path / "output.png"

    runner = CliRunner()
    result = runner.invoke(main, ["viz", str(input_path), "-o", str(output_path), "--shape", "50x100"])
    assert result.exit_code == 0
    assert output_path.exists()


def test_cli_process_unsupported_format(tmp_path: Path):
    input_path = tmp_path / "test.xyz"
    input_path.write_text("dummy")
    output_path = tmp_path / "output.stl"

    runner = CliRunner()
    result = runner.invoke(main, ["process", str(input_path), "-o", str(output_path)])
    assert result.exit_code != 0
    assert "Unsupported file format" in result.output
