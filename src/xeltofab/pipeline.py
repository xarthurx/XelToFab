"""Pipeline orchestrator."""

from __future__ import annotations

from xeltofab.extract import extract
from xeltofab.preprocess import preprocess
from xeltofab.smooth import smooth
from xeltofab.state import PipelineState


def process(state: PipelineState) -> PipelineState:
    """Run the full preprocessing -> extraction -> smoothing pipeline."""
    state = preprocess(state)
    state = extract(state)
    state = smooth(state)
    return state
