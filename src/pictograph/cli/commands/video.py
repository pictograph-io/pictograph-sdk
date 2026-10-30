"""``pictograph video {upload,probe,extract-frames,status}``.

Turn a video file into an annotatable image dataset in three steps::

    pictograph video upload clip.mp4                       # prints the video handle
    pictograph video probe <video>                         # duration / fps / dimensions
    pictograph video extract-frames my-dataset <video> --directory /clips/run-1 --sample-fps 2

``upload`` sends a local video to temporary storage; the ``video`` it prints feeds both
``probe`` and ``extract-frames``. ``extract-frames``
is asynchronous on the backend; by default it polls until the job finishes
(status ``complete``) and prints the populated row. Pass ``--no-wait`` to
fire-and-forget and poll later via ``video status``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pictograph.cli._client import get_client
from pictograph.cli._errors import command_group
from pictograph.cli._format import print_json, print_table

app = command_group(no_args_is_help=True)


@app.command("upload", help="Upload a local video to temporary storage (prints its video handle).")
def upload_video(
    file: Annotated[Path, typer.Argument(help="Local video file to upload.")],
    content_type: Annotated[
        str,
        typer.Option(
            "--content-type",
            help="MIME type. Set for non-MP4 sources (video/quicktime, video/webm, …).",
        ),
    ] = "video/mp4",
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if not file.is_file():
        raise typer.BadParameter(f"video file not found: {file}", param_hint="FILE")
    client = get_client(api_key)
    info = client.video.upload(file, content_type=content_type)
    print_json(info.model_dump(mode="json", exclude_none=True))


@app.command("probe", help="Probe an uploaded video for duration / fps / dimensions.")
def probe_video(
    video: Annotated[str, typer.Argument(help="The video handle `video upload` printed.")],
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    meta = client.video.probe(video)
    if json_output:
        print_json(meta.model_dump(mode="json", exclude_none=True))
        return
    rows = [
        {
            "duration_seconds": round(meta.duration_seconds, 2),
            "native_fps": round(meta.native_fps, 3),
            "width": meta.width,
            "height": meta.height,
            "frame_count": meta.frame_count,
        }
    ]
    print_table(rows, title="Video metadata")


@app.command(
    "extract-frames", help="Extract frames into a dataset directory; waits unless --no-wait."
)
def extract_frames(
    dataset: Annotated[str, typer.Argument(help="Dataset name within your org.")],
    video: Annotated[str, typer.Argument(help="The video handle `video upload` printed.")],
    directory: Annotated[
        str,
        typer.Option(
            "--directory", help="Dataset directory the frames go into, e.g. /clips/run-1."
        ),
    ],
    sample_fps: Annotated[
        float, typer.Option("--sample-fps", help="Frames per second to extract.")
    ] = 1.0,
    wait: Annotated[
        bool, typer.Option("--wait/--no-wait", help="Poll until the extraction job completes.")
    ] = True,
    poll_interval: Annotated[
        float, typer.Option("--poll-interval", help="Seconds between polls.")
    ] = 3.0,
    timeout: Annotated[float, typer.Option("--timeout", help="Max seconds to wait.")] = 1800.0,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    job = client.video.extract_frames(
        dataset,
        video,
        directory=directory,
        sample_fps=sample_fps,
        wait=wait,
        poll_interval=poll_interval,
        timeout=timeout,
    )
    print_json(job.model_dump(mode="json", exclude_none=True))


@app.command("status", help="Poll a frame-extraction job's status + progress.")
def extraction_status(
    job: Annotated[str, typer.Argument(help="Frame-extraction job id.")],
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    state = client.video.get_extraction(job)
    if json_output:
        print_json(state.model_dump(mode="json", exclude_none=True))
        return
    rows = [
        {
            "job_id": state.job_id,
            "status": state.status,
            "progress": f"{state.progress}%",
            "frames": f"{state.frames_extracted}/{state.total_frames}",
            "directory_path": state.directory_path,
            "error": state.error,
        }
    ]
    print_table(rows, title="Frame extraction")
