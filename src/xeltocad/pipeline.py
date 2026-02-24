"""Pipeline orchestrator."""

from __future__ import annotations

from xeltocad.extract import extract
from xeltocad.preprocess import preprocess
from xeltocad.smooth import smooth
from xeltocad.state import PipelineState


def process(state: PipelineState) -> PipelineState:
    """Run the full preprocessing -> extraction -> smoothing pipeline."""
    state = preprocess(state)
    state = extract(state)
    state = smooth(state)
    return state
