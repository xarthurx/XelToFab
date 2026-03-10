"""Pipeline state and parameter models."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PipelineParams(BaseModel):
    """Configurable parameters for the pipeline."""

    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    smooth_sigma: float = Field(default=1.0, ge=0.0)
    morph_radius: int = Field(default=1, ge=0)
    smoothing_method: Literal["taubin", "bilateral"] = "taubin"
    taubin_iterations: int = Field(default=20, ge=0)
    taubin_lambda: float = Field(default=0.5, gt=0.0, le=1.0)  # shrinkage factor for Taubin smoothing
    bilateral_iterations: int = Field(default=10, ge=1)
    bilateral_sigma_s: float | None = Field(default=None, gt=0.0)  # auto from avg edge length
    bilateral_sigma_n: float = Field(default=0.35, gt=0.0)  # normal similarity threshold (radians)

    field_type: Literal["density", "sdf"] = "density"
    direct_extraction: bool = False
    extraction_level: float | None = None

    # Mesh repair (3D only, requires pymeshlab)
    repair: bool = True

    # Isotropic remeshing (3D only, requires gpytoolbox)
    remesh: bool = True
    target_edge_length: float | None = Field(default=None, gt=0.0)
    remesh_iterations: int = Field(default=10, ge=1)

    @model_validator(mode="after")
    def apply_field_type_defaults(self) -> PipelineParams:
        """Apply smart defaults based on field_type.

        Only override values that were not explicitly set by the user.
        SDF defaults: direct_extraction=True, smooth_sigma=0.0 (Gaussian preprocessing only;
        Taubin mesh smoothing still runs since marching cubes produces staircase artifacts).
        """
        if self.field_type == "sdf":
            explicitly_set = self.model_fields_set
            if "direct_extraction" not in explicitly_set:
                self.direct_extraction = True
            if "smooth_sigma" not in explicitly_set:
                self.smooth_sigma = 0.0
        return self

    @property
    def effective_extraction_level(self) -> float:
        """Extraction level: explicit value, or 0.0 for SDF, threshold for density."""
        if self.extraction_level is not None:
            return self.extraction_level
        return 0.0 if self.field_type == "sdf" else self.threshold


class PipelineState(BaseModel):
    """Pipeline state threaded through stage functions.

    Stage functions use model_copy(update={...}) to return new state objects.
    Note: model_copy is a shallow copy — numpy arrays are shared between copies.
    Do NOT mutate arrays in-place; always create new arrays in stage functions.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    field: np.ndarray
    ndim: int = 0  # computed from field
    params: PipelineParams = Field(default_factory=PipelineParams)

    binary: np.ndarray | None = None
    vertices: np.ndarray | None = None
    faces: np.ndarray | None = None
    contours: list[np.ndarray] | None = None
    smoothed_vertices: np.ndarray | None = None
    volume_fraction: float | None = None

    @field_validator("field")
    @classmethod
    def validate_field(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim not in (2, 3):
            raise ValueError(f"field must be 2D or 3D, got {v.ndim}D")
        return v

    @property
    def best_vertices(self) -> np.ndarray | None:
        """Return smoothed vertices if available, otherwise raw vertices."""
        return self.smoothed_vertices if self.smoothed_vertices is not None else self.vertices

    @model_validator(mode="after")
    def set_ndim(self) -> PipelineState:
        self.ndim = self.field.ndim
        return self
