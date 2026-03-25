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
| `heat_conduction_3d_51x51x51_sample{0,1}.npy` | (51, 51, 51) | `IDEALLab/heat_conduction_3d_v0` |

## 3D Thermoelastic

3D topology optimization for coupled thermoelastic problems (smaller resolution).

| File | Shape | Source Dataset |
|------|-------|----------------|
| `thermoelastic_3d_16x16x16_sample{0,1}.npy` | (16, 16, 16) | `IDEALLab/thermoelastic_3d_v0` |

## 3D Corner-Based (Structural)

3D SIMP topology optimization results for structural compliance problems.
Source: [Corner-Based TO Dataset](https://github.com/dustin-bielecki/Corner-Based-Topology-Optimization-Dataset)
(Bielecki et al., "Multi-stage deep neural network accelerated topology optimization", Structural and Multidisciplinary Optimization).

| File | Shape | Volume Fraction | Source |
|------|-------|-----------------|--------|
| `corner_3d_20x20x20_vf50_sample0.npy` | (20, 20, 20) | 50% | `3D/outputs_50_1.mat` sample 0 |
| `corner_3d_20x20x20_vf30_sample0.npy` | (20, 20, 20) | 30% | `3D/outputs_30_1.mat` sample 0 |

## Usage

```python
from xeltofab.io import load_field
from xeltofab.pipeline import process

state = load_field("data/examples/beams_2d_50x100_sample0.npy")
result = process(state)
```
