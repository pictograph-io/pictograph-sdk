"""Every task has its typed overload (ROADMAP Program N, 2026-11-25): ``Deployments.connect`` and ``DeploymentClient``
narrow ``task="..."`` to that task's result type. Anomaly detection, the sixth task, had none: a caller passing
``task="anomaly_detection"`` matched no overload (mypy: no overload variant) though the call worked. Read off the
overloads themselves, so a seventh task cannot miss them either."""

from __future__ import annotations

import sys
import typing
from typing import get_args

import pytest

from pictograph.inference.results import TASK_RESULT_TYPES, TaskName
from pictograph.resources._deployment_client import DeploymentClient
from pictograph.resources.deployments import Deployments

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="typing.get_overloads is Python 3.11+"
)


def _task_overloads(
    fn: object, result_of: typing.Callable[[dict[str, str]], str]
) -> dict[str, str]:
    out: dict[str, str] = {}
    for o in typing.get_overloads(fn):  # type: ignore[arg-type]
        ann = o.__annotations__
        task = ann.get("task", "")
        if task.startswith("Literal["):
            out[task[len('Literal["') : -len('"]')]] = result_of(ann)
    return out


@pytest.mark.parametrize("task", get_args(TaskName))
def test_connect_narrows_every_task(task: str) -> None:
    got = _task_overloads(Deployments.connect, lambda ann: ann["return"])
    assert got.get(task) == f"DeploymentClient[{TASK_RESULT_TYPES[task].__name__}]"


@pytest.mark.parametrize("task", get_args(TaskName))
def test_the_client_narrows_every_task(task: str) -> None:
    got = _task_overloads(DeploymentClient.__init__, lambda ann: ann["self"])
    assert got.get(task) == f"DeploymentClient[{TASK_RESULT_TYPES[task].__name__}]"
