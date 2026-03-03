# Notebooks

Interactive notebooks built with [marimo](https://marimo.io/) — a reactive Python notebook that runs as a `.py` file.

## Available Notebooks

| Notebook | Description |
|----------|-------------|
| `demo.py` | End-to-end pipeline demo: density field preprocessing, mesh extraction, and Taubin smoothing |
| `demo_formats.py` | Round-trip showcase of all supported I/O formats (`.npy`, `.mat`, `.vtk`, `.h5`, etc.) |

## Running

Make sure dependencies are installed first:

```bash
uv sync
```

**Launch a notebook in the browser (edit mode):**

```bash
uv run marimo edit notebooks/demo.py
```

**View a notebook as a read-only app:**

```bash
uv run marimo run notebooks/demo.py
```

marimo can also convert notebooks to HTML or run them headlessly — see `uv run marimo --help` for all options.
