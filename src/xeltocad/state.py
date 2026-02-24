"""Pipeline state and parameter models."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PipelineParams(BaseModel):
    """Configurable parameters for the pipeline."""

    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    smooth_sigma: float = Field(default=1.0, ge=0.0)
    morph_radius: int = Field(default=1, ge=0)
    taubin_iterations: int = Field(default=20, ge=0)
    taubin_pass_band: float = Field(default=0.1, gt=0.0, le=1.0)


class PipelineState(BaseModel):
    """Immutable pipeline state threaded through stage functions."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    density: np.ndarray
    ndim: int = 0  # computed from density
    params: PipelineParams = Field(default_factory=PipelineParams)

    binary: np.ndarray | None = None
    vertices: np.ndarray | None = None
    faces: np.ndarray | None = None
    contours: list[np.ndarray] | None = None
    smoothed_vertices: np.ndarray | None = None
    volume_fraction: float | None = None

    @field_validator("density")
    @classmethod
    def validate_density(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim not in (2, 3):
            raise ValueError(f"density must be 2D or 3D, got {v.ndim}D")
        return v

    @model_validator(mode="after")
    def set_ndim(self) -> PipelineState:
        self.ndim = self.density.ndim
        return self
