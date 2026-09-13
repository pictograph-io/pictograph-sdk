# Contributing

Thanks for helping improve the Pictograph Python SDK.

## How changes land

This repository is published from Pictograph's internal source tree. `main` is
replaced wholesale on each release, so a merge commit here would not survive the
next one.

That does not make contributions second-class - it just changes the mechanics:

- **Issues are the fastest path.** A clear bug report with a minimal reproduction
  gets fixed and released, usually within a release or two.
- **Pull requests are welcome and are read.** A maintainer applies the change to
  the canonical tree with your authorship preserved in the changelog entry, then
  closes the PR referencing the release that carries it. Your PR being *closed*
  rather than *merged* is the normal, successful outcome here.
- Small, focused changes land fastest. If you are planning something large, open
  an issue first so we can agree on the shape before you write it.

## Setup

```bash
git clone https://github.com/pictograph-io/pictograph-sdk.git
cd pictograph-sdk
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,cli,agents]"
```

Requires Python 3.10 or newer.

## The quality gate

All four must pass. CI runs the same set on every pull request, across Python
3.10 through 3.13.

```bash
ruff check src/pictograph/ tests/ examples/          # lint
ruff format --check src/pictograph/ tests/ examples/
mypy src/pictograph/ && mypy examples/               # strict
pytest                                               # unit + integration
```

`pytest` never touches the network. Live tests are deselected by default and
require an explicit opt-in - see below.

## Conventions

- **No `Any`.** `mypy --strict` is the floor. Reach for `Annotated[..., Field(...)]`,
  discriminated unions, or a specific protocol.
- **Pydantic at every boundary.** Request models set `extra="forbid"`; response
  models set `extra="ignore"` so a new server field never breaks an old client.
- **The annotation label field is `name`, never `class`.** Enforced throughout.
- **Every method has an async twin** with the same signature under
  `pictograph.aio`. Change both, or neither.
- **Tests assert an invariant or a failure mode.** "It imports" is not a test. If
  a test fails, fix the code rather than the assertion.
- Add a bullet under `## [Unreleased]` in `CHANGELOG.md` describing the change in
  terms of what a user can now do differently. Do not edit `_version.py`; releases
  are cut by a maintainer.

## Live tests spend real money

`tests/live/` runs against the production API. It creates real datasets and
consumes real credits, including GPU time on the auto-annotation and training
paths.

They are gated twice so this cannot happen by accident: the `live` and `e2e`
markers are deselected by default, **and** the key must be `PICTOGRAPH_TEST_KEY`.
`PICTOGRAPH_API_KEY` deliberately does not unlock them, because that is the
variable you have exported for ordinary SDK use.

```bash
PICTOGRAPH_TEST_KEY=pk_live_... pytest tests/live/ -m live
```

Use a key for a throwaway organization, never a production one.

## Reporting bugs

Open an issue with the SDK version, the Python version, and the smallest snippet
that reproduces the problem. Redact your API key - a key in an issue is public
the moment you press submit, and should be rotated immediately.

For anything security-related, do **not** open a public issue. Follow
[SECURITY.md](SECURITY.md).
