# Example Density Fields

Pre-computed topology optimization results for testing and demos. All files are numpy `.npy` arrays with continuous density values in [0, 1].

## Sources

Data sourced from [IDEALLab EngiBench](https://huggingface.co/IDEALLab) HuggingFace datasets.

## 2D Beams

Cantilever beam optimization results at three resolutions. Each sample has different load/volume fraction conditions.

| File | Shape | Source Dataset |
|------|-------|----------------|
| `beams_2d_25x50_sample{0,1,2}.npy` | (25, 50) | `IDEALLab/beams_2d_25_50_v0` |
| `beams_2d_50x100_sample{0,1,2}.npy` | (50, 100) | `IDEALLab/beams_2d_50_100_v0` |
| `beams_2d_100x200_sample{0,1,2}.npy` | (100, 200) | `IDEALLab/beams_2d_100_200_v0` |

## 3D Heat Conduction

3D topology optimization for heat conduction problems.

| File | Shape | Source Dataset |
|------|-------|----------------|
| `heat_conduction_3d_51x51x51_sample{0,1,2}.npy` | (51, 51, 51) | `IDEALLab/heat_conduction_3d_v0` |

## 3D Thermoelastic

3D topology optimization for coupled thermoelastic problems (smaller resolution).

| File | Shape | Source Dataset |
|------|-------|----------------|
| `thermoelastic_3d_16x16x16_sample{0,1}.npy` | (16, 16, 16) | `IDEALLab/thermoelastic_3d_v0` |

## Usage

```python
from xeltocad.io import load_density
from xeltocad.pipeline import process

state = load_density("examples/data/beams_2d_50x100_sample0.npy")
result = process(state)
```
