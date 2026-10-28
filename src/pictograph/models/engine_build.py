"""The TensorRT engine builds a workflow run or a deployment waits on.

The first time a model runs on a GPU at a precision, PRT builds its TensorRT engine (a minute
or two for most models) and stores it; every later workflow run or deployment of that model on
that GPU at that precision loads the stored engine in seconds. A workflow run reports its builds
on :attr:`~pictograph.models.workflow.WorkflowRun.engine_build`; a provisioning deployment reports
the build it waits on on :attr:`~pictograph.models.deployment.Deployment.engine_build`, and a
workflow's live endpoint on :attr:`~pictograph.models.workflow.WorkflowEndpoint.engine_build`. A
build is never billed.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class EngineBuildModel(BaseModel):
    """One model's TensorRT engine build: which model, at what precision, and how far along.

    The first run of a model on a GPU at a precision - a workflow run, or a deployment
    provisioning - builds its engine first, then it is stored and later ones on that GPU start
    in seconds. The build is not billed.
    """

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    model_id: str
    name: str | None = None
    precision: str
    # "building" or "built" - a string, so a new state never breaks an older SDK.
    state: str
    # When this build started (UTC): how long it has run so far.
    started_at: datetime | None = None
    # How long builds like it took on that GPU before; None when there is no history.
    estimate_s: float | None = None
    # How long it took, once built.
    build_s: float | None = None


class EngineBuild(BaseModel):
    """The TensorRT engines a workflow run or a deployment builds or built: ``state`` is
    ``"building"`` while one is under way and ``"built"`` once all have finished. Nothing to
    build (every engine was already stored) reads as ``None`` on the run or the deployment."""

    model_config = ConfigDict(extra="ignore")

    state: str
    models: list[EngineBuildModel] = Field(default_factory=list)
