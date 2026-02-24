# tests/test_cli.py
from pathlib import Path

import numpy as np
from click.testing import CliRunner

from xeltocad.cli import main


def test_cli_process_3d(tmp_path: Path):
    # Create test input
    z, y, x = np.mgrid[-1:1:20j, -1:1:20j, -1:1:20j]
    density = (x**2 + y**2 + z**2 < 0.5**2).astype(float)
    input_path = tmp_path / "sphere.npy"
    np.save(input_path, density)
    output_path = tmp_path / "sphere.stl"

    runner = CliRunner()
    result = runner.invoke(main, ["process", str(input_path), "-o", str(output_path)])
    assert result.exit_code == 0
    assert output_path.exists()


def test_cli_process_with_params(tmp_path: Path):
    z, y, x = np.mgrid[-1:1:20j, -1:1:20j, -1:1:20j]
    density = (x**2 + y**2 + z**2 < 0.5**2).astype(float)
    input_path = tmp_path / "sphere.npy"
    np.save(input_path, density)
    output_path = tmp_path / "sphere.stl"

    runner = CliRunner()
    result = runner.invoke(
        main, ["process", str(input_path), "-o", str(output_path), "--threshold", "0.4", "--sigma", "1.5"]
    )
    assert result.exit_code == 0
    assert output_path.exists()


def test_cli_viz_2d(tmp_path: Path):
    y, x = np.mgrid[-1:1:50j, -1:1:50j]
    density = (x**2 + y**2 < 0.5**2).astype(float)
    input_path = tmp_path / "circle.npy"
    np.save(input_path, density)
    output_path = tmp_path / "circle.png"

    runner = CliRunner()
    result = runner.invoke(main, ["viz", str(input_path), "-o", str(output_path)])
    assert result.exit_code == 0
    assert output_path.exists()
