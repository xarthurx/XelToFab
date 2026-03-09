"""Pipeline orchestrator."""

from __future__ import annotations

from xeltofab.extract import extract
from xeltofab.preprocess import preprocess
from xeltofab.smooth import smooth
from xeltofab.state import PipelineState


def process(state: PipelineState) -> PipelineState:
    """Run the pipeline: [preprocess] -> extraction -> smoothing.

    When direct_extraction is enabled, preprocessing is skipped and extraction
    operates on the continuous input field directly.
    """
    if not state.params.direct_extraction:
        state = preprocess(state)
    state = extract(state)
    state = smooth(state)
    return state
