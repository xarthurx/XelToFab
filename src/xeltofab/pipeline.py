"""Pipeline orchestrator."""

from __future__ import annotations

from xeltofab.decimate import decimate
from xeltofab.extract import extract
from xeltofab.preprocess import preprocess
from xeltofab.remesh import remesh
from xeltofab.repair import repair
from xeltofab.smooth import smooth
from xeltofab.state import PipelineState


def process(state: PipelineState) -> PipelineState:
    """Run the pipeline: [preprocess] -> extract -> smooth -> [repair] -> [remesh] -> [decimate].

    When direct_extraction is enabled, preprocessing is skipped and extraction
    operates on the continuous input field directly.
    Repair, remesh, and decimate are enabled by default.
    """
    if not state.params.direct_extraction:
        state = preprocess(state)
    state = extract(state)
    state = smooth(state)
    if state.params.needs_repair:
        state = repair(state)
    state = remesh(state)
    state = decimate(state)
    return state
