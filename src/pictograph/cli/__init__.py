"""Pictograph CLI - invokable as ``pictograph`` after install.

The CLI mirrors the SDK 1:1: every command maps to one or more SDK
calls. Typer + Rich are optional deps gated under the ``[cli]`` extra::

    pip install 'pictograph[cli]'

The entry point ``pictograph = pictograph.cli._app:main`` lives in
``pyproject.toml``. ``main()`` wraps the Typer ``app`` so an expected
``PictographError`` (401/402/404/409/429) renders as a one-line
``error: …`` instead of a Rich traceback. Importing ``pictograph.cli``
directly is rare - users invoke the CLI binary, not the Python module.
"""
