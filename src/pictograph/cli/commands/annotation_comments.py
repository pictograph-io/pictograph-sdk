"""``pictograph annotation-comments {list,add,reply,edit,resolve,delete}`` - notes on annotations.

How an agent and a person talk about a label: the agent leaves a note on the annotation
it is unsure of, the person answers or resolves it in the app (or here). ::

    pictograph annotation-comments list road-signs 001.jpg --unresolved
    pictograph annotation-comments add road-signs 001.jpg <annotation-id> "Is this a yield sign?"
    pictograph annotation-comments reply road-signs 001.jpg <comment-id> "Yes - relabelled."
    pictograph annotation-comments resolve <comment-id>

An annotation is addressed by its id (``pictograph annotations get <dataset> <image>``
prints them): a class name is not unique on an image. ``@username`` in a comment
notifies that member of the organization.
"""

from __future__ import annotations

import sys
from typing import Annotated, Any

import typer

from pictograph.cli._client import get_client
from pictograph.cli._errors import CliError, ExitCode, command_group
from pictograph.cli._format import print_json, print_table

app = command_group(no_args_is_help=True)

IMAGE_HELP = (
    "The image: its filename, or its id (a filename that is in more than one folder "
    "needs the id; pictograph images list prints it)."
)
BODY_HELP = "The comment ('-' reads it from stdin). @username notifies that member."


def _body(text: str) -> str:
    body = sys.stdin.read() if text == "-" else text
    if not body.strip():
        raise typer.BadParameter("the comment is empty.", param_hint="BODY")
    return body


def _dump(comment: Any) -> dict[str, Any]:
    out: dict[str, Any] = comment.model_dump(mode="json", exclude_none=True)
    return out


@app.command("list", help="Every comment on an image's annotations, oldest first.")
def list_comments(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    image: Annotated[str, typer.Argument(help=IMAGE_HELP)],
    unresolved: Annotated[
        bool, typer.Option("--unresolved", help="Only the threads not resolved yet.")
    ] = False,
    json_output: Annotated[
        bool, typer.Option("--json", help="Print the comments as JSON.")
    ] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    comments = [c for c in client.annotation_comments.list(dataset, image) if not c.is_deleted]
    if unresolved:
        open_threads = {c.id for c in comments if c.parent_comment_id is None and not c.resolved}
        comments = [c for c in comments if (c.parent_comment_id or c.id) in open_threads]
    if json_output:
        print_json([_dump(c) for c in comments])
        return
    rows = [
        {
            "id": c.id,
            "annotation": c.annotation_id,
            "reply_to": c.parent_comment_id,
            "author": c.author_username or c.author_name,
            "comment": c.body if len(c.body) <= 80 else c.body[:79] + "…",
            "resolved": c.resolved,
        }
        for c in comments
    ]
    print_table(rows, title=f"Comments on {image} ({len(rows)})")


@app.command("add", help="Comment on one annotation of an image.")
def add_comment(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    image: Annotated[str, typer.Argument(help=IMAGE_HELP)],
    annotation_id: Annotated[
        str,
        typer.Argument(help="The annotation's id (pictograph annotations get prints them)."),
    ],
    body: Annotated[str, typer.Argument(help=BODY_HELP)],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    text = _body(body)
    client = get_client(api_key)
    comment = client.annotation_comments.create(dataset, image, annotation=annotation_id, body=text)
    print_json(_dump(comment))


@app.command("reply", help="Reply to a comment on an image (its thread).")
def reply_comment(
    dataset: Annotated[str, typer.Argument(help="Dataset name.")],
    image: Annotated[str, typer.Argument(help=IMAGE_HELP)],
    comment: Annotated[str, typer.Argument(help="The comment to reply to.")],
    body: Annotated[str, typer.Argument(help=BODY_HELP)],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    text = _body(body)
    client = get_client(api_key)
    parent = next(
        (c for c in client.annotation_comments.list(dataset, image) if c.id == comment), None
    )
    if parent is None:
        raise CliError(
            f"Image {image!r} of {dataset!r} has no comment {comment!r}.",
            exit_code=ExitCode.NOT_FOUND,
            code="comment_not_found",
            fix=f"List them with `pictograph annotation-comments list {dataset} {image}`.",
        )
    result = client.annotation_comments.reply(
        dataset,
        image,
        annotation=parent.annotation_id,
        parent=comment,
        body=text,
    )
    print_json(_dump(result))


@app.command("edit", help="Change the text of a comment you wrote.")
def edit_comment(
    comment: Annotated[str, typer.Argument(help="The comment.")],
    body: Annotated[str, typer.Argument(help=BODY_HELP)],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    text = _body(body)
    client = get_client(api_key)
    print_json(_dump(client.annotation_comments.update(comment, body=text)))


@app.command("resolve", help="Mark a comment's thread resolved (--reopen opens it again).")
def resolve_comment(
    comment: Annotated[str, typer.Argument(help="The comment.")],
    reopen: Annotated[bool, typer.Option("--reopen", help="Open the thread again.")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    result = client.annotation_comments.resolve(comment, resolved=not reopen)
    print_json(_dump(result))


@app.command(
    "delete",
    help="Delete a comment (yours, or any as an org admin). It keeps its place in the "
    "thread with an empty body.",
)
def delete_comment(
    comment: Annotated[str, typer.Argument(help="The comment.")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation.")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if not yes and not typer.confirm(f"Delete comment {comment!r}?"):
        raise typer.Abort()
    client = get_client(api_key)
    client.annotation_comments.delete(comment)
    print_json({"comment_id": comment, "deleted": True})
