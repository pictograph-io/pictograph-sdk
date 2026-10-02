"""``pictograph organizations {me,members,member-role,member-remove,invites,invite,invite-revoke}``.

Inspect and manage the calling key's organization - tier, members, and invites.
Everything is implicitly scoped to the API key's org; there is no
``organization_id`` flag anywhere (cross-org access is impossible by construction)::

    pictograph organizations me                                   # tier + credit balance + member cap
    pictograph organizations members                              # who's on the team
    pictograph organizations invite alice@acme.com --role admin   # onboard a colleague
    pictograph organizations invites --status pending             # see outstanding invites

Member and invite *mutations* require ``admin`` or ``owner`` role on the key.
"""

from __future__ import annotations

from typing import Annotated

import typer

from pictograph.cli._client import get_client
from pictograph.cli._format import print_json, print_table

app = typer.Typer(no_args_is_help=True)


@app.command("me", help="Show the calling key's organization (tier, credits, member cap).")
def me(
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    organization = client.organizations.me()
    print_json(organization.model_dump(mode="json", exclude_none=True))


@app.command("update", help="Update your organization's profile (name / description / public).")
def update(
    name: Annotated[str | None, typer.Option("--name", help="New organization name.")] = None,
    description: Annotated[
        str | None, typer.Option("--description", "-d", help="Public profile description.")
    ] = None,
    is_public: Annotated[
        bool | None,
        typer.Option("--public/--private", help="Show or hide the org's public profile page."),
    ] = None,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    organization = client.organizations.update(
        name=name, description=description, is_public=is_public
    )
    print_json(organization.model_dump(mode="json", exclude_none=True))


@app.command("members", help="List every member of the organization with role and email.")
def members(
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    members = client.organizations.list_members()
    if json_output:
        print_json([m.model_dump(mode="json", exclude_none=True) for m in members])
        return
    rows = [
        {
            "id": m.id,
            "email": m.email,
            "name": m.full_name,
            "role": m.role,
            "joined_at": m.joined_at,
        }
        for m in members
    ]
    print_table(rows, title=f"Members ({len(rows)})")


@app.command(
    "member-role", help="Change a member's role (admin/owner only; can't demote last owner)."
)
def member_role(
    member_id: Annotated[str, typer.Argument(help="Membership row UUID (from `members`).")],
    role: Annotated[str, typer.Option("--role", help="owner / admin / member / viewer.")],
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    member = client.organizations.update_member_role(member_id, role=role)  # type: ignore[arg-type]
    print_json(member)


@app.command("member-remove", help="Remove a member from the org (admin/owner only).")
def member_remove(
    member_id: Annotated[str, typer.Argument(help="Membership row UUID (from `members`).")],
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if not yes and not typer.confirm(f"Remove member {member_id!r} from the organization?"):
        raise typer.Abort()
    client = get_client(api_key)
    client.organizations.remove_member(member_id)
    print_json({"id": member_id, "removed": True})


@app.command("invites", help="List invites in the org, optionally filtered by status.")
def invites(
    status: Annotated[
        str | None, typer.Option("--status", help="pending/accepted/expired/revoked.")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    invites = client.organizations.list_invites(status=status)  # type: ignore[arg-type]
    if json_output:
        print_json([i.model_dump(mode="json", exclude_none=True) for i in invites])
        return
    rows = [
        {
            "id": i.id,
            "email": i.email,
            "role": i.role,
            "status": i.status,
            "expires_at": i.expires_at,
        }
        for i in invites
    ]
    print_table(rows, title=f"Invites ({len(rows)})")


@app.command("invite", help="Invite a new member by email (admin/owner only; sends an email).")
def invite(
    email: Annotated[str, typer.Argument(help="Address to invite (lowercased server-side).")],
    role: Annotated[
        str, typer.Option("--role", help="admin / member (default) / viewer.")
    ] = "member",
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    client = get_client(api_key)
    created = client.organizations.invite(email, role=role)  # type: ignore[arg-type]
    print_json(created.model_dump(mode="json", exclude_none=True))


@app.command("invite-revoke", help="Revoke a pending invite (admin/owner only).")
def invite_revoke(
    invite_id: Annotated[str, typer.Argument(help="Invite UUID (from `invites`).")],
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
) -> None:
    if not yes and not typer.confirm(f"Revoke invite {invite_id!r}?"):
        raise typer.Abort()
    client = get_client(api_key)
    client.organizations.revoke_invite(invite_id)
    print_json({"id": invite_id, "revoked": True})
