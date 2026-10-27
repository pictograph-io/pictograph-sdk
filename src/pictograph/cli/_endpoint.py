"""ONE live endpoint at the terminal: what ``pictograph deployments`` and ``pictograph workflows`` share (B579).

Where an endpoint is served and its autoscaling (the options every endpoint command takes, said once), its token's
one showing, and its streams: :func:`run_stream` opens a stream on an endpoint's direct client (a deployment's, or a
workflow's - the same client), says where to watch it (and where to send its video), prints each frame - and with
``--all`` every message - as one JSON line, and says why it ended. Ctrl-C stops it.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Annotated, Any

import typer

from pictograph.cli._format import err_console, print_json_line

if TYPE_CHECKING:
    from pictograph.models.deployment import DeploymentRuntime
    from pictograph.models.engine_build import EngineBuild

# ── where it is served and its autoscaling: the options every endpoint command takes ──

REGION_HELP = (
    "Where it is served: us-east or us-west (`pictograph deployments compute-options` lists them), or auto "
    "(default) - the region nearest where this runs. Pick the one nearest your callers; it never changes once "
    "deployed."
)
MIN_HELP = "Containers kept warm (0 = scale to zero, billed only while one is up)."
MAX_HELP = "Autoscale ceiling."
SCALEDOWN_HELP = "Idle seconds before scale-down."

# Bounds beyond these are the server's to refuse (its SCALING_LIMITS), so they are said in one place.
Region = Annotated[str, typer.Option("--region", help=REGION_HELP)]
MinContainers = Annotated[int, typer.Option("--min-containers", "--min", min=0, help=MIN_HELP)]
MaxContainers = Annotated[int, typer.Option("--max-containers", "--max", min=1, help=MAX_HELP)]
ScaledownWindow = Annotated[
    int, typer.Option("--scaledown-window", "--scaledown", help=SCALEDOWN_HELP)
]
# The same, for a change: only what is given changes.
MinChange = Annotated[int | None, typer.Option("--min-containers", "--min", min=0, help=MIN_HELP)]
MaxChange = Annotated[int | None, typer.Option("--max-containers", "--max", min=1, help=MAX_HELP)]
ScaledownChange = Annotated[
    int | None, typer.Option("--scaledown-window", "--scaledown", help=SCALEDOWN_HELP)
]


def say_token_once(where: str) -> None:
    """The warning beside an endpoint token's one showing: ``where`` names what a member reveals it from in the app
    ("the deployment's Analytics", "the workflow's Analytics")."""
    err_console.print(
        "[bold yellow]Save this token now - the API won't show it again "
        f"(a member can reveal it from {where} in the app).[/bold yellow]"
    )


# ── what it is doing: its state and its engines' build, in a person's words ──


def is_building(build: EngineBuild | None) -> bool:
    """Whether an endpoint waits on a TensorRT engine being built."""
    return build is not None and build.state == "building"


def build_estimate_s(build: EngineBuild | None) -> float | None:
    """How long builds like the one an endpoint waits on took before (None: no history)."""
    estimates = [m.estimate_s for m in (build.models if build else []) if m.estimate_s is not None]
    return max(estimates) if estimates else None


def build_state(build: EngineBuild | None) -> str:
    """An endpoint's engine build, in a few words ("" when there is none)."""
    if build is None:
        return ""
    if is_building(build):
        estimate = build_estimate_s(build)
        return f"building (about {estimate:.0f}s)" if estimate is not None else "building"
    times = [m.build_s for m in build.models if m.build_s is not None]
    return f"built in {sum(times):.0f}s" if times else build.state


def waited(runtime: DeploymentRuntime) -> str:
    """How long a request has waited for a container ("" when the wait has no start), on the server's clock when it
    says the time it read it."""
    if runtime.waiting_since is None:
        return ""
    now = runtime.checked_at or datetime.now(timezone.utc)
    seconds = max(0, int((now - runtime.waiting_since).total_seconds()))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def live_status(status: str, runtime: DeploymentRuntime | None) -> str:
    """The status a person reads: an ``active`` endpoint with what it is doing right now - ``active · live (2)``,
    ``active · asleep``, ``active · waiting for a GPU 1m42s``."""
    if status != "active" or runtime is None:
        return status
    if runtime.state == "live":
        starting = f", {runtime.creating} starting" if runtime.creating else ""
        live = f"live ({runtime.containers}{starting})"
    elif runtime.state == "starting":
        live = f"starting ({max(runtime.creating, 1)})"
    elif runtime.state == "waiting":
        live = " ".join(filter(None, ("waiting for a GPU", waited(runtime))))
    else:  # asleep, unknown, or a state added later: its own word
        live = runtime.state
    return f"{status} · {live}"


def say_waiting(name: str, build: EngineBuild | None, *, building: str, until: str) -> None:
    """What a ``--wait`` says it waits on: ``building`` (the engines, in words) while they are built, else the endpoint
    reaching ``until``."""
    if is_building(build):
        estimate = build_estimate_s(build)
        about = f" (about {estimate:.0f}s)" if estimate is not None else ""
        err_console.print(f"Building {building} first{about}, then starting {name}...")
    else:
        err_console.print(f"Waiting for {name} to {until}...")


# ── its streams ──

#: Why a stream ended, as the CLI says it (``DeploymentStream.ended``).
STREAM_ENDED = {
    "source_ended": "its camera stopped sending",
    "idle": "nothing read it or sent to it for 30 seconds",
    "time_limit": "it reached its four hours",
    "container_stopped": "its container stopped",
    "ended": "it had already ended",
}


def run_stream(
    conn: Any,
    source: str,
    *,
    overlay: bool,
    frames: int | None,
    seconds: float | None,
    all_messages: bool,
) -> int:
    """Stream ``source`` through ``conn`` (a :class:`~pictograph.DeploymentClient`) until its end, Ctrl-C, ``frames``
    frames or ``seconds`` seconds: the frames printed."""
    started = time.monotonic()
    printed = 0
    with conn, conn.open_stream(source, overlay=overlay) as live:
        # soft_wrap: a URL printed at the terminal's width must stay one line to copy.
        err_console.print(
            f"Streaming ({live.id}). Watch it at {live.whep_url} (WHEP, with the token).",
            soft_wrap=True,
        )
        if live.whip_url:
            err_console.print(
                f"Send its video to {live.whip_url} (WHIP, H.264, with the token).",
                soft_wrap=True,
            )
        err_console.print("Ctrl-C stops it.")
        try:
            for message in live.messages():
                is_frame = message.get("type") == "frame"
                if is_frame or all_messages:
                    print_json_line(message)
                printed += int(is_frame)
                if (frames is not None and printed >= frames) or (
                    seconds is not None and time.monotonic() - started >= seconds
                ):
                    break
        except KeyboardInterrupt:
            pass
        ended = live.ended
    said = STREAM_ENDED.get(ended or "")
    tail = f": {said}" if said else ""
    err_console.print(
        f"The stream ended after {printed:,} frame{'' if printed == 1 else 's'}{tail}.",
        soft_wrap=True,
    )
    return printed
