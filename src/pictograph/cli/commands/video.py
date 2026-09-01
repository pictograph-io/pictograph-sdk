"""``pictograph video {upload,probe,extract-frames,status}``.

Turn a video file into an annotatable image dataset in three steps::

    pictograph video upload clip.mp4                       # prints the temporary gcs_path
    pictograph video probe <gcs_path>                      # duration / fps / dimensions
    pictograph video extract-frames my-dataset <gcs_path> --directory-name clip --sample-fps 2

``upload`` streams a local video straight to a temporary storage path; the returned
``gcs_path`` feeds both ``probe`` and ``extract-frames``. ``extract-frames``
is asynchronous on the backend; by default it polls until the job finishes
(status ``complete``) and prints the populated row. Pass ``--no-wait`` to
fire-and-forget and poll later via ``video status``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pictograph.cli._client import get_client
from pictograph.cli._format import print_json, print_table

app = typer.Typer(no_args_is_help=True)


@app.command("upload", help="Upload a local video to temporary storage (prints the gcs_path).")
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
    gcs_path: Annotated[str, typer.Argument(help="The gcs_path returned from `video upload`.")],
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    meta = client.video.probe(gcs_path)
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
    dataset_name: Annotated[str, typer.Argument(help="Dataset name within your org.")],
    gcs_path: Annotated[str, typer.Argument(help="The gcs_path returned from `video upload`.")],
    directory_name: Annotated[
        str, typer.Option("--directory-name", help="Virtual directory to create for the frames.")
    ],
    sample_fps: Annotated[
        float, typer.Option("--sample-fps", help="Frames per second to extract.")
    ] = 1.0,
    parent_directory_path: Annotated[
        str, typer.Option("--parent-directory-path", help="Parent directory in the dataset.")
    ] = "/",
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
        dataset_name,
        gcs_path,
        directory_name=directory_name,
        sample_fps=sample_fps,
        parent_directory_path=parent_directory_path,
        wait=wait,
        poll_interval=poll_interval,
        timeout=timeout,
    )
    print_json(job.model_dump(mode="json", exclude_none=True))


@app.command("status", help="Poll a frame-extraction job's status + progress.")
def extraction_status(
    job_id: Annotated[str, typer.Argument(help="Frame-extraction job id.")],
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    job = client.video.get_extraction(job_id)
    if json_output:
        print_json(job.model_dump(mode="json", exclude_none=True))
        return
    rows = [
        {
            "job_id": job.job_id,
            "status": job.status,
            "progress": f"{job.progress}%",
            "frames": f"{job.frames_extracted}/{job.total_frames}",
            "directory_path": job.directory_path,
            "error": job.error,
        }
    ]
    print_table(rows, title="Frame extraction")
