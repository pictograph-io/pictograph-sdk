# ------------------------------------------------------------------------
# Pictograph - original work, not vendored.
# ------------------------------------------------------------------------
"""The RF-DETR architecture, vendored from rfdetr 1.8.3 (Apache-2.0). See NOTICE.

A trained RF-DETR checkpoint is a state dict: to run it natively you must first
rebuild the exact `nn.Module` it was trained as. Upstream that means installing the
`rfdetr` package - which, to hand back an architecture, also brings `transformers`
(pinned `>=5.1,<6`), `supervision`, `pydeprecate` and their resolver constraints
into the user's environment. The SDK does not ask for that. The model-construction
subset lives here instead, so `pip install pictograph` is the whole requirement.

What is vendored is the BUILD path only - backbone, transformer, heads, the weight
loader and the postprocessor. Training (loss, matcher, datasets, the Lightning
stack), export, LoRA and the Roboflow platform integrations are not.

**The version is load-bearing.** These modules are 1.8.3, matching
the training service's `rfdetr==1.8.3` pin,
because the weights we ship encode that architecture. Re-vendoring from a different
release without retraining is how a checkpoint starts loading cleanly and predicting
nonsense. Re-sync both together.
"""

from __future__ import annotations

from pictograph.inference._rfdetr.builder import Detections, RFDETRModel, from_checkpoint

__all__ = ["Detections", "RFDETRModel", "from_checkpoint"]

#: The upstream release this tree was vendored from.
RFDETR_VENDORED_VERSION = "1.8.3"
