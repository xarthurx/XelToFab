# Contributing to XelToFab

Thank you for your interest in contributing to XelToFab! This document explains how to get involved.

## Reporting Bugs

Open an issue at <https://github.com/xarthurx/XelToFab/issues> with:

- A clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Python version and OS

## Suggesting Features

Open an issue with the **enhancement** label. Describe the use case and, if possible, link to relevant literature or examples.

## Development Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/xarthurx/XelToFab.git
   cd XelToFab
   ```

2. Install [uv](https://docs.astral.sh/uv/) if you don't have it:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. Install the project with dev dependencies:

   ```bash
   uv sync --group dev
   ```

4. (Optional) Install extra backends:

   ```bash
   uv sync --group dev --extra all
   ```

## Running Tests

```bash
uv run pytest tests/ -v
```

## Code Style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check     # lint
ruff format    # format
```

Please run both before submitting a pull request.

## Pull Requests

1. Fork the repository and create a feature branch from `main`.
2. Keep changes focused — one logical change per PR.
3. Add or update tests for any new functionality.
4. Make sure all tests pass and `ruff check` is clean.
5. Write a clear PR description explaining *what* and *why*.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
