# ------------------------------------------------------------------------
# Pictograph - original work, not vendored.
# ------------------------------------------------------------------------
"""The two `rfdetr.assets.model_weights` entry points `weights.py` calls, minus the network.

Upstream, these resolve a variant name (``rf-detr-nano.pth``) against Roboflow's
asset registry and fetch it over HTTP with an MD5 check. The SDK never wants that:
the checkpoint is always a concrete local file the caller already downloaded from
their own organization's storage, and silently reaching out to a third-party CDN
mid-load - potentially overwriting that file - is exactly the behaviour we are
removing along with the dependency.

So the download becomes an existence assertion. A path that is present is used
as-is; a path that is not is an error naming the file, rather than a fetch.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

__all__ = ["download_pretrain_weights", "get_model_cache_dir", "validate_pretrain_weights"]


def get_model_cache_dir() -> str:
    """Where a BARE checkpoint filename would resolve to.

    Only reachable from `ModelConfig.expand_path` when a config field carries a
    filename with no directory component - which, for us, is the variant class's
    own unused default (`rf-detr-nano.pth`), never the checkpoint we load. Honours
    upstream's `RF_HOME` so an existing cache is still found, and keeps its default
    location for the same reason. Nothing is ever written here.
    """
    return str(Path(os.environ.get("RF_HOME") or "~/.roboflow/models").expanduser())


def download_pretrain_weights(pretrain_weights: str, **_kwargs: Any) -> None:
    """Assert the checkpoint is on disk. Never fetches.

    Raises:
        FileNotFoundError: if `pretrain_weights` does not name an existing file.
    """
    if Path(pretrain_weights).is_file():
        return
    raise FileNotFoundError(
        f"RF-DETR checkpoint {pretrain_weights!r} does not exist. The Pictograph SDK "
        f"loads weights only from a local file that it downloaded from your "
        f"organization's model storage; it never fetches pretrained weights from a "
        f"third-party host."
    )


def validate_pretrain_weights(
    pretrain_weights: str,  # noqa: ARG001 - signature matches the upstream call site
    strict: bool = True,  # noqa: ARG001
    **_kwargs: Any,
) -> None:
    """No-op. Upstream validates the name + MD5 against the remote asset registry,
    which does not apply to a checkpoint produced by our own training pipelines."""
    return
