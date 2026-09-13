import cv2
import numpy as np
import onnxruntime as ort

from .semantic_calibration import semantic_probabilities


class SemanticSegmentationModelPyTorch:
    def __init__(
        self,
        model_path: str,
        classes: list[str] = [],
        input_shape: tuple[int] = (512, 512),
        class_confidences: dict[int, float] = {},
        confidence_threshold: float = 0.5,
        providers: list[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"],
        sess_options: ort.SessionOptions | None = None,
    ):
        self.model_path: str = model_path
        self.dims: tuple[int] = input_shape
        self.ratio: float = 1.0
        self.classes: list[str] = classes
        self.providers: list[str] = providers
        self.class_confidences: dict[int, float] = class_confidences
        # The per-class `class_confidences` dict overrides this for any class
        # it names; every other class falls back to this single threshold -
        # previously always a hardcoded 0.5 regardless of what the caller
        # passed, so `confidence_threshold` from `build_wrapper()` was silently
        # ignored for this whole model family.
        self.confidence_threshold: float = confidence_threshold

        if sess_options is None:
            sess_options = ort.SessionOptions()

        self.session = ort.InferenceSession(
            self.model_path,
            providers=self.providers,
            sess_options=sess_options,
        )

    def preprocess(self, image: np.ndarray, bgr2rgb: bool = True):
        """
        Standard ImageNet preprocessing:
        1. Convert BGR to RGB
        2. Resize to model input dimensions
        3. Normalize to [0, 1] by dividing by 255
        4. Apply ImageNet mean/std normalization
        5. Transpose from (H, W, C) to (C, H, W) for PyTorch

        Args:
                image: Input image in BGR format, uint8, values [0, 255]
                bgr2rgb: Whether to convert BGR to RGB

        Returns:
                Preprocessed image (C, H, W) with normalized values
        """
        if bgr2rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.dims[1], self.dims[0]))  # dims=(H,W); cv2 dsize=(W,H)

        # Convert to float32 and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std

        # Transpose from (H, W, C) to (C, H, W)
        image = np.transpose(image, (2, 0, 1))
        return image

    def postprocess(self, prediction: np.ndarray, return_probs: bool = False):
        """Convert the model's RAW output to binary masks: calibrate, argmax, per-class gate.

        The calibration step is not optional. `create_model` bakes a sigmoid in
        for a SINGLE-class Unet/UnetPlusPlus and for nothing else - every
        multi-class model, and every Segformer, emits raw unbounded logits - so
        thresholding `prediction` directly compared a logit against a
        PROBABILITY-scale confidence (0.5 by default). `semantic_probabilities`
        is the one shared place that question is answered; the torch engine
        (dispatch.semantic_masks_from_logits) calls the same function, so both
        backends gate identically. See semantic_calibration.py for the
        measurement and for the detection's failure mode.

        Args:
                prediction: the model's raw output, already upscaled to the original
                        resolution by `predict`.
                return_probs: OPT-IN. When True, also return the calibrated
                        probability maps that produced the masks, aligned 1:1 with them
                        (`probs[i]` is the probability map of the class `masks[i]`
                        covers). They are the evidence behind each emitted region, and
                        `dispatch._semantic_seg_to_annotations` turns them into a real
                        per-region `confidence` instead of the pydantic default of 1.0.
                        Default False so every existing caller - `run_job`'s
                        `_run_semantic_seg`, the per-deployment app's warmup, the
                        workflow runner - keeps its single-value return unchanged.
                        DELIBERATELY a return value, not wrapper state: these wrappers
                        already carry per-call state (`self.ratio`) that forces callers
                        to serialize, and stashing the maps would make that worse.

        Returns:
                The `(num_classes, H, W)` binary mask stack, or `(masks, probs)`
                when `return_probs` is set.
        """
        # Single-channel output (e.g. prediction.shape == (H, W))
        if prediction.ndim == 2:
            # channel_axis=None: this array HAS no channel axis, which cannot be
            # inferred from ndim (a (C, N) stack is 2-D too).
            probabilities = semantic_probabilities(prediction, channel_axis=None)
            # For binary, use class_id 0's confidence threshold
            threshold = self.class_confidences.get(0, self.confidence_threshold)
            single_mask = (probabilities >= threshold).astype(np.uint8)
            stacked = np.expand_dims(single_mask, axis=0)
            return (stacked, [probabilities]) if return_probs else stacked

        # Multi-channel output (e.g. prediction.shape == (H, W, C))
        if prediction.ndim == 3:
            probabilities = semantic_probabilities(prediction, channel_axis=2)

            # The per-class loop below indexes `probabilities[:, :, class_idx]`
            # for class_idx in 1..len(classes), so it ASSUMES the graph emitted
            # exactly `len(classes) + 1` channels (background + one per class).
            # When the stored class_mapping drifts from the served ONNX graph
            # (a model retrained with a different class count, or a stale
            # registry list) that assumption breaks two ways, both silent:
            #   - FEWER channels than classes -> `probabilities[:, :, class_idx]`
            #     raised a bare `IndexError` mid-loop, 500-ing the whole request
            #     with an opaque message (B457).
            #   - MORE channels than classes -> the loop bound silently DROPPED
            #     the trailing channel(s), under-emitting real segmentation classes.
            # Either way the positional class<->channel alignment the emitter
            # relies on is broken, so the masks would be MISLABELLED. Fail loudly
            # with an actionable message instead of emitting garbage or crashing
            # opaquely. A correctly-configured model never hits this (its channel
            # count == len(class_mapping) + 1 by construction), so real inference
            # is byte-identical.
            n_channels = probabilities.shape[2]
            n_expected = len(self.classes) + 1
            if n_channels != n_expected:
                raise ValueError(
                    f"Semantic-seg channel/class mismatch: the ONNX graph emitted "
                    f"{n_channels} channel(s) (background + {max(0, n_channels - 1)} "
                    f"class channel(s)) but the model's class_mapping declares "
                    f"{len(self.classes)} class(es) (expected {n_expected} channels). "
                    f"The registry class list has drifted from the served graph; "
                    f"re-sync the model's class_mapping to the ONNX output."
                )

            # Standard semantic segmentation: assign each pixel to the class with highest probability
            # prediction has shape (H, W, num_classes+1) where channel 0 is background

            # Use argmax to get the class with highest probability per pixel.
            # (Unchanged by calibration - softmax is monotonic - but taken on the
            # calibrated array so there is exactly one source of truth here.)
            class_predictions = np.argmax(probabilities, axis=2)  # Shape: (H, W)

            # Post-process filtering - for each class, check if that class own probability meets its confidence threshold
            # Create binary masks for each class (excluding background which is class 0)
            masks: list[np.ndarray] = []
            probs: list[np.ndarray] = []
            for class_idx in range(1, len(self.classes) + 1):  # Start from 1 to skip background
                # Find pixels assigned to this class by argmax
                assigned_to_class = class_predictions == class_idx

                # Get confidence threshold for this specific class using class_id (0-indexed)
                class_id = class_idx - 1  # -1 because class_idx starts at 1 (0 is background)
                threshold = self.class_confidences.get(class_id, self.confidence_threshold)
                class_probabilities = probabilities[:, :, class_idx]
                meets_threshold = class_probabilities >= threshold

                # Final mask: pixels assigned to this class and meets threshold
                mask_i = (assigned_to_class & meets_threshold).astype(np.uint8)
                masks.append(mask_i)
                probs.append(class_probabilities)
            stacked = np.stack(masks, axis=0)
            return (stacked, probs) if return_probs else stacked

        raise ValueError(f"Unsupported prediction shape: {prediction.shape}")

    def predict(
        self,
        image: np.ndarray,
        preprocess: bool = True,
        postprocess: bool = True,
        return_probs: bool = False,
    ):
        """One BGR image -> the `(num_classes, H, W)` binary mask stack.

        `return_probs=True` additionally returns the calibrated per-class
        probability maps behind those masks - see `postprocess`. It is an opt-in
        precisely so the existing callers' `masks = wrapper.predict(img)` keeps
        working untouched.
        """
        if return_probs and not postprocess:
            # There are no probability maps without the calibrate+gate step, and
            # silently returning a differently-shaped result for one flag
            # combination is exactly how a caller ends up unpacking raw logits.
            raise ValueError("return_probs=True requires postprocess=True")

        height_orig, width_orig, channels = image.shape
        if preprocess:
            image = self.preprocess(image=image)
        # ONNX inference - output will be (1, C, H, W)
        input_name = self.session.get_inputs()[0].name
        onnx_pred = self.session.run(None, {input_name: np.expand_dims(image, axis=0)})[0]
        # Remove batch dimension: (1, C, H, W) -> (C, H, W)
        onnx_pred = onnx_pred.squeeze(0)
        # Transpose to (H, W, C) or squeeze to (H, W) for single channel
        if onnx_pred.shape[0] == 1:  # Single class: (1, H, W) -> (H, W)
            onnx_pred = onnx_pred.squeeze(0)
        else:  # Multiple classes: (C, H, W) -> (H, W, C)
            onnx_pred = np.transpose(onnx_pred, (1, 2, 0))

        if postprocess:
            # For multiclass, resize each channel separately to preserve probability distributions
            if onnx_pred.ndim == 3:
                # Resize each channel independently using linear interpolation
                resized_channels = []
                for i in range(onnx_pred.shape[2]):
                    channel = cv2.resize(
                        onnx_pred[:, :, i],
                        (width_orig, height_orig),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    resized_channels.append(channel)
                onnx_pred = np.stack(resized_channels, axis=2)
            else:
                # Binary case: simple resize
                onnx_pred = cv2.resize(
                    onnx_pred, (width_orig, height_orig), interpolation=cv2.INTER_LINEAR
                )

            return self.postprocess(onnx_pred, return_probs=return_probs)
        return onnx_pred
