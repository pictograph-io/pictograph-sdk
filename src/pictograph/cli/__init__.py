"""Pictograph CLI - invokable as ``pictograph`` after install.

The CLI mirrors the SDK 1:1: every command maps to one or more SDK
calls. Typer + Rich are optional deps gated under the ``[cli]`` extra::

    pip install 'pictograph[cli]'

The entry point ``pictograph = pictograph.cli._app:main`` lives in
``pyproject.toml``. Every expected failure renders as a one-line ``error: …``
(or, with ``--json`` / ``PICTOGRAPH_OUTPUT=json``, one line of JSON on stderr)
and exits with one code per kind of failure - auth 2, forbidden 3, not found 4,
conflict 5, invalid input 6, credits 7, rate limited 8, unavailable 9, a wait
that ran out 10 (``pictograph.cli._errors``). Importing ``pictograph.cli``
directly is rare - users invoke the CLI binary, not the Python module.
"""
