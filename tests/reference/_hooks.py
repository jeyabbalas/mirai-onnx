"""Forward-hook + monkey-patch helpers for Phase 0 tensor capture.

Each install_* function mutates a shared `captured` dict via side effects and
returns a `cleanup()` callable that restores upstream behavior. Always pair with
`try / finally cleanup()`.

The capture targets are documented in the Phase 0 plan; in short:
- DICOM raw uint16 -> module-attribute monkey-patch on the relevant decoder.
- Preproc tensor -> wrap MiraiModel.collate_batch.
- image_encoder_out / xai_hidden / raw_logit / risk_factor_vector / per-key pred RFs
  -> forward hooks attached at process_image_joint entry (after model is loaded).
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np
import torch
from PIL import Image


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    # Force a fresh allocation: hooks capture tensors that may be mutated
    # in-place by upstream code (e.g. ReLU in aggregate_and_classify).
    return t.detach().cpu().contiguous().numpy().copy()


def install_dicom_capture(captured: Dict[str, Any], use_dcmtk: bool) -> Callable[[], None]:
    """Monkey-patch the active DICOM decoder so we capture both the per-image
    uint16 array AND the (view, side) order in which slots are filled."""
    import onconet.utils.dicom as dmod
    import pydicom

    captured["dicom_raw_uint16"] = {}
    captured["_batch_slot_to_view_side"] = []

    if use_dcmtk:
        original = dmod.dicom_to_image_dcmtk

        def wrapper(dicom_path, image_path):
            pil = original(dicom_path, image_path)
            pil.load()  # force eager pixel read; the upstream temp file may vanish soon
            arr = np.array(pil)
            if arr.dtype != np.uint16:
                arr = arr.astype(np.uint16)
            dcm = pydicom.dcmread(dicom_path, stop_before_pixels=True)
            view, side = dmod.get_dicom_info(dcm)
            captured["dicom_raw_uint16"][(view, side)] = arr.copy()
            captured["_batch_slot_to_view_side"].append((view, side))
            return pil

        dmod.dicom_to_image_dcmtk = wrapper

        def cleanup() -> None:
            dmod.dicom_to_image_dcmtk = original

    else:
        original = dmod.dicom_to_arr

        def wrapper(dicom, window_method="minmax", index=0, pillow=False, overlay=False):
            # Capture by re-calling the original with pillow=False (no extra
            # decode work — pixel_array is cached on the dataset).
            arr = original(
                dicom,
                window_method=window_method,
                index=index,
                pillow=False,
                overlay=overlay,
            )
            view, side = dmod.get_dicom_info(dicom)
            captured["dicom_raw_uint16"][(view, side)] = arr.copy()
            captured["_batch_slot_to_view_side"].append((view, side))
            if pillow:
                image = arr.astype(np.int32)
                if image.shape[-1] in {3, 4}:
                    image = image.mean(axis=-1, dtype=np.int32)
                return Image.fromarray(image, mode="I")
            return arr

        dmod.dicom_to_arr = wrapper

        def cleanup() -> None:
            dmod.dicom_to_arr = original

    return cleanup


def install_collate_capture(model_holder, captured: Dict[str, Any]) -> Callable[[], None]:
    """Wrap MiraiModel.collate_batch to save per-image preproc tensors."""
    original = model_holder.collate_batch
    captured["preproc_tensor"] = {}

    def wrapper(images, transforms):
        batch = original(images, transforms)
        x = batch["x"]  # (1, C, N, H, W) after the .transpose(1, 2) in collate_batch
        n = x.shape[2]
        assert n == len(images), (n, len(images))
        for slot in range(n):
            view = int(images[slot]["view_seq"])
            side = int(images[slot]["side_seq"])
            tensor_chw = _to_numpy(x[0, :, slot, :, :])
            captured["preproc_tensor"][(view, side)] = tensor_chw
        captured["preproc_batch_x_shape"] = tuple(int(s) for s in x.shape)
        captured["preproc_batch_x_dtype"] = str(x.dtype)
        return batch

    model_holder.collate_batch = wrapper

    def cleanup() -> None:
        model_holder.collate_batch = original

    return cleanup


def install_post_load_hooks(model_holder, captured: Dict[str, Any]) -> Callable[[], None]:
    """Wrap process_image_joint so we can attach forward hooks on the loaded
    model right before inference runs (and tear them down right after)."""
    original = model_holder.process_image_joint
    outer_handles: list = []

    def wrapper(batch, model, calibrator, risk_factor_vector=None):
        # Record the actual training mode of every key module. Upstream Mirai never
        # calls .eval() in MiraiModel.run_model — inference runs in TRAIN mode (the
        # snapshot only restores weights, not mode). We match upstream behavior.
        captured["module_training_mode"] = {
            type(m).__name__: bool(m.training)
            for m in [
                model,
                model.image_encoder,
                model.transformer,
                model.transformer.pool,
                model.transformer.prob_of_failure_layer,
            ]
        }
        assert "raw_logit" not in captured, "process_image_joint already captured once"

        rf_dim = int(model.transformer.pool.length_risk_factor_vector)
        captured["rf_dim"] = rf_dim
        captured["risk_factor_keys"] = list(model.transformer.pool.args.risk_factor_keys)
        captured["risk_factor_key_to_num_class"] = dict(
            model.transformer.pool.args.risk_factor_key_to_num_class
        )

        handles: list = []

        def pre_transformer(module, inputs):
            img_x = inputs[0]
            captured["image_encoder_out"] = _to_numpy(img_x)

        handles.append(model.transformer.register_forward_pre_hook(pre_transformer))

        def post_pool(module, inputs, output):
            # output is (None, hidden) where hidden = [image_hidden ; rf_hidden].
            _, hidden = output
            h = _to_numpy(hidden)
            captured["pool_hidden"] = h
            captured["image_hidden_in_pool"] = h[:, :-rf_dim].copy()
            captured["risk_factor_vector"] = h[:, -rf_dim:].copy()

        handles.append(model.transformer.pool.register_forward_hook(post_pool))

        # Each per-key FC is called TWICE during one inference pass:
        #   1. RiskFactorPool.forward, with the raw internal_pool hidden
        #      (this is what feeds the concatenated risk_factor_vector)
        #   2. AllImageTransformer.forward -> self.pool.get_pred_rf_loss(...),
        #      with hidden post-self.relu (in-place ReLU on the pool output).
        # Only the first call is what gets concatenated; capture it and ignore the second.
        captured["pred_risk_factor_logits_per_key"] = {}

        def make_hook(k):
            def hook(module, inputs, output):
                if k not in captured["pred_risk_factor_logits_per_key"]:
                    captured["pred_risk_factor_logits_per_key"][k] = _to_numpy(output)

            return hook

        for key in model.transformer.pool.args.risk_factor_keys:
            fc = model.transformer.pool._modules[f"{key}_fc"]
            handles.append(fc.register_forward_hook(make_hook(key)))

        def pre_pof(module, inputs):
            (h,) = inputs
            captured["xai_hidden"] = _to_numpy(h)

        handles.append(
            model.transformer.prob_of_failure_layer.register_forward_pre_hook(pre_pof)
        )

        def post_pof(module, inputs, output):
            captured["raw_logit"] = _to_numpy(output)

        handles.append(
            model.transformer.prob_of_failure_layer.register_forward_hook(post_pof)
        )

        try:
            return original(batch, model, calibrator, risk_factor_vector)
        finally:
            for h in handles:
                h.remove()

    model_holder.process_image_joint = wrapper

    def cleanup() -> None:
        model_holder.process_image_joint = original
        for h in outer_handles:
            h.remove()

    return cleanup


def derive_pred_risk_factors_per_key(captured: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """From per-key logits, derive per-key probability tensors (sigmoid for binary
    keys, softmax for multi-class). Mirrors RiskFactorPool.forward lines 49-52."""
    out: Dict[str, np.ndarray] = {}
    n_class_map = captured["risk_factor_key_to_num_class"]
    for key, logit in captured["pred_risk_factor_logits_per_key"].items():
        n_class = n_class_map[key]
        if n_class == 1:
            out[key] = 1.0 / (1.0 + np.exp(-logit))  # sigmoid
        else:
            shifted = logit - logit.max(axis=-1, keepdims=True)
            ex = np.exp(shifted)
            out[key] = ex / ex.sum(axis=-1, keepdims=True)
    return out
