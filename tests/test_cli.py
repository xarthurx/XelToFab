# tests/test_cli.py
from pathlib import Path

import numpy as np
from click.testing import CliRunner

from xeltofab.cli import main


def test_cli_process_3d(tmp_path: Path, small_sphere_field: np.ndarray):
    input_path = tmp_path / "sphere.npy"
    np.save(input_path, small_sphere_field)
    output_path = tmp_path / "sphere.stl"

    runner = CliRunner()
    result = runner.invoke(main, ["process", str(input_path), "-o", str(output_path)])
    assert result.exit_code == 0
    assert output_path.exists()


def test_cli_process_with_params(tmp_path: Path, small_sphere_field: np.ndarray):
    input_path = tmp_path / "sphere.npy"
    np.save(input_path, small_sphere_field)
    output_path = tmp_path / "sphere.stl"

    runner = CliRunner()
    result = runner.invoke(
        main, ["process", str(input_path), "-o", str(output_path), "--threshold", "0.4", "--sigma", "1.5"]
    )
    assert result.exit_code == 0
    assert output_path.exists()


def test_cli_viz_2d(tmp_path: Path, circle_field: np.ndarray):
    input_path = tmp_path / "circle.npy"
    np.save(input_path, circle_field)
    output_path = tmp_path / "circle.png"

    runner = CliRunner()
    result = runner.invoke(main, ["viz", str(input_path), "-o", str(output_path)])
    assert result.exit_code == 0
    assert output_path.exists()


def test_cli_process_2d_shows_error(tmp_path: Path, circle_field: np.ndarray):
    """CLI should show friendly error for 2D process (no mesh export)."""
    input_path = tmp_path / "circle.npy"
    np.save(input_path, circle_field)
    output_path = tmp_path / "circle.stl"

    runner = CliRunner()
    result = runner.invoke(main, ["process", str(input_path), "-o", str(output_path)])
    assert result.exit_code != 0
    assert "2D contour export" in result.output


def test_cli_process_sdf(tmp_path: Path, small_sphere_sdf: np.ndarray):
    """CLI process with --field-type sdf."""
    input_path = tmp_path / "sphere_sdf.npy"
    np.save(input_path, small_sphere_sdf)
    output_path = tmp_path / "sphere.stl"

    runner = CliRunner()
    result = runner.invoke(main, ["process", str(input_path), "-o", str(output_path), "--field-type", "sdf"])
    assert result.exit_code == 0, result.output
    assert output_path.exists()


def test_cli_process_direct(tmp_path: Path, small_sphere_field: np.ndarray):
    """CLI process with --direct flag for clean density."""
    input_path = tmp_path / "sphere.npy"
    np.save(input_path, small_sphere_field)
    output_path = tmp_path / "sphere.stl"

    runner = CliRunner()
    result = runner.invoke(main, ["process", str(input_path), "-o", str(output_path), "--direct"])
    assert result.exit_code == 0, result.output
    assert output_path.exists()


def test_cli_process_no_repair(tmp_path: Path, small_sphere_field: np.ndarray):
    """CLI process with --no-repair flag."""
    input_path = tmp_path / "sphere.npy"
    np.save(input_path, small_sphere_field)
    output_path = tmp_path / "sphere.stl"

    runner = CliRunner()
    result = runner.invoke(main, ["process", str(input_path), "-o", str(output_path), "--no-repair"])
    assert result.exit_code == 0, result.output
    assert output_path.exists()


def test_cli_process_no_remesh(tmp_path: Path, small_sphere_field: np.ndarray):
    """CLI process with --no-remesh flag."""
    input_path = tmp_path / "sphere.npy"
    np.save(input_path, small_sphere_field)
    output_path = tmp_path / "sphere.stl"

    runner = CliRunner()
    result = runner.invoke(main, ["process", str(input_path), "-o", str(output_path), "--no-remesh"])
    assert result.exit_code == 0, result.output
    assert output_path.exists()
