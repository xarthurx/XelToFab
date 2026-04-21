"""XelToFab — Design fields to fabrication-ready geometry."""

from xeltofab.convert import (
    heaviside,
    linear_ramp,
    sdf_to_density,
    sigmoid,
)
from xeltofab.extrude import extrude_2d
from xeltofab.io import (
    load_field,
    save_mesh,
)
from xeltofab.pipeline import process, process_from_sdf
from xeltofab.state import PipelineParams, PipelineState

__all__ = [
    "PipelineParams",
    "PipelineState",
    "extrude_2d",
    "heaviside",
    "linear_ramp",
    "load_field",
    "process",
    "process_from_sdf",
    "save_mesh",
    "sdf_to_density",
    "sigmoid",
]
