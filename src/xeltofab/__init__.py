"""XelToFab — Design fields to fabrication-ready geometry."""

from xeltofab.io import (
    load_density,  # deprecated alias — use load_field()
    load_field,
    save_mesh,
)
from xeltofab.pipeline import process
from xeltofab.state import PipelineParams, PipelineState

__all__ = [
    "PipelineParams",
    "PipelineState",
    "load_field",
    "load_density",  # deprecated alias
    "process",
    "save_mesh",
]
