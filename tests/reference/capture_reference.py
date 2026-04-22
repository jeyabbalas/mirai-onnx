"""Phase 0 capture: run upstream Mirai once on the four demo DICOMs (twice,
once via pydicom and once via dcmtk) and freeze every intermediate tensor we
care about under tests/reference/fixtures/.

Default behavior refuses to overwrite existing fixtures; pass --regenerate to
opt into a clean rewrite.

Usage:
    python -m tests.reference.capture_reference                # write if missing
    python -m tests.reference.capture_reference --regenerate   # overwrite
    python -m tests.reference.capture_reference --only pydicom # one path only
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import os
import pathlib
import pickle
import platform
import random
import shutil
import socket
import subprocess
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

# Project-relative paths.
HERE = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
MIRAI_ROOT = PROJECT_ROOT / "external" / "Mirai"
MIRAI_CONFIG = MIRAI_ROOT / "onconet" / "configs" / "mirai_trained.json"
DEMO_DATA_DIR = PROJECT_ROOT / "mirai_demo_data"
DEMO_FILES = ["ccl1.dcm", "ccr1.dcm", "mlol2.dcm", "mlor2.dcm"]
FIXTURES_DIR = HERE / "fixtures"
PREVIEW_DIR = FIXTURES_DIR / "preview"

VIEW_STR = {0: "CC", 1: "MLO"}
SIDE_STR = {0: "R", 1: "L"}


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def set_determinism() -> Dict[str, Any]:
    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)  # torch >= 1.11
        det_mode = "warn_only"
    except TypeError:
        # torch 1.9 doesn't support warn_only kwarg; the api is just (mode: bool).
        try:
            torch.use_deterministic_algorithms(True)
            det_mode = "strict"
        except RuntimeError:
            det_mode = "unsupported"
    return {
        "torch_set_num_threads": 1,
        "use_deterministic_algorithms": det_mode,
        "seeds": {"python": 0, "numpy": 0, "torch": 0, "PYTHONHASHSEED": "0"},
    }


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------
def _build_dicom_byteios() -> List[io.BytesIO]:
    out = []
    for name in DEMO_FILES:
        with open(DEMO_DATA_DIR / name, "rb") as f:
            out.append(io.BytesIO(f.read()))
    return out


def run_pipeline_with_hooks(use_dcmtk: bool) -> Dict[str, Any]:
    """Run the upstream Mirai pipeline once with all hooks attached.

    Returns the captured-tensor dict, plus derived `predictions`, `raw_sigmoid`,
    and `calibrated` arrays. Asserts cross-checks pass before returning.
    """
    from onconet.predict import _load_config
    from onconet.models.mirai_full import MiraiModel

    from tests.reference._hooks import (
        derive_pred_risk_factors_per_key,
        install_collate_capture,
        install_dicom_capture,
        install_post_load_hooks,
    )

    config = _load_config(str(MIRAI_CONFIG), threads=1)
    config.cuda = False
    model_holder = MiraiModel(config)

    captured: Dict[str, Any] = {}
    cleanups: List = []
    cleanups.append(install_dicom_capture(captured, use_dcmtk=use_dcmtk))
    cleanups.append(install_collate_capture(model_holder, captured))
    cleanups.append(install_post_load_hooks(model_holder, captured))

    try:
        dicom_data = _build_dicom_byteios()
        payload = {"dcmtk": use_dcmtk, "window_method": "minmax"}
        report = model_holder.run_model(dicom_data, payload=payload)
        captured["predictions"] = report["predictions"]
        captured["modelVersion"] = report.get("modelVersion") or model_holder.__version__
    finally:
        for cb in cleanups:
            try:
                cb()
            except Exception:
                pass

    # Derived: raw_sigmoid (numpy), calibrated (numpy), pred_risk_factors_per_key (dict).
    raw_logit = captured["raw_logit"]
    captured["raw_sigmoid"] = (1.0 / (1.0 + np.exp(-raw_logit))).astype(np.float32)
    captured["pred_risk_factors_per_key"] = derive_pred_risk_factors_per_key(captured)
    captured["calibrated"] = _apply_calibrator(
        captured["raw_sigmoid"],
        os.path.expanduser(config.calibrator_path),
    )

    _run_self_checks(captured)
    return captured


def _apply_calibrator(raw_sigmoid: np.ndarray, calibrator_path: str) -> np.ndarray:
    with open(calibrator_path, "rb") as f:
        calibrator = pickle.load(f)
    n_years = raw_sigmoid.shape[1]
    out = np.zeros(n_years, dtype=np.float64)
    for i in calibrator.keys():
        out[i] = calibrator[i].predict_proba(
            raw_sigmoid[0, i].reshape(-1, 1)
        ).flatten()[1]
    return out


def _run_self_checks(captured: Dict[str, Any]) -> None:
    """Cross-checks that must pass before we trust the capture."""
    rf_dim = captured["rf_dim"]

    # 4 unique (view, side) tuples in the slot order list.
    order = captured["_batch_slot_to_view_side"]
    assert len(order) == 4, f"expected 4 dicom decodes, got {len(order)}: {order}"
    assert len(set(order)) == 4, f"duplicate (view, side): {order}"

    # No NaN/Inf in any captured fp tensor.
    for name in (
        "image_encoder_out",
        "pool_hidden",
        "image_hidden_in_pool",
        "risk_factor_vector",
        "xai_hidden",
        "raw_logit",
        "raw_sigmoid",
        "calibrated",
    ):
        arr = captured[name]
        assert np.isfinite(arr).all(), f"{name} has NaN/Inf"

    # Hidden split: xai_hidden = relu(pool_hidden) within FP tolerance.
    relu_pool = np.maximum(captured["pool_hidden"], 0.0)
    np.testing.assert_allclose(
        captured["xai_hidden"], relu_pool, atol=1e-6, rtol=0,
        err_msg="xai_hidden != relu(pool_hidden); upstream relu path changed",
    )

    # The trailing rf_dim slice of xai_hidden must equal relu(risk_factor_vector).
    np.testing.assert_allclose(
        captured["xai_hidden"][:, -rf_dim:],
        np.maximum(captured["risk_factor_vector"], 0.0),
        atol=1e-6, rtol=0,
        err_msg="xai_hidden tail does not match relu(risk_factor_vector)",
    )

    # Concatenation of per-key probs (in risk_factor_keys order) must equal risk_factor_vector.
    per_key = captured["pred_risk_factors_per_key"]
    cat = np.concatenate([per_key[k] for k in captured["risk_factor_keys"]], axis=1)
    np.testing.assert_allclose(
        cat, captured["risk_factor_vector"], atol=1e-6, rtol=0,
        err_msg="per-key probs do not concat to risk_factor_vector",
    )

    # Round(calibrated, 4) must equal report's printed predictions.
    pred = captured["predictions"]
    for i, p in enumerate(captured["calibrated"]):
        upstream = pred[f"Year {i + 1}"]
        assert round(float(p), 4) == upstream, (
            f"Year {i+1}: round(calibrated,4)={round(float(p),4)} vs upstream={upstream}"
        )

    # raw_logit non-negative monotone-cumulative (cumulative hazards).
    rl = captured["raw_logit"][0]
    assert all(rl[i] <= rl[i + 1] + 1e-6 for i in range(len(rl) - 1)), (
        f"raw_logit not monotone-nondecreasing: {rl.tolist()}"
    )


# ---------------------------------------------------------------------------
# Fixture writers
# ---------------------------------------------------------------------------
def _suffix(use_dcmtk: bool) -> str:
    return "_dcmtk" if use_dcmtk else ""


def write_fixtures(captured: Dict[str, Any], use_dcmtk: bool) -> Dict[str, Dict]:
    """Write every captured tensor to disk; return a {relpath: info} dict for the manifest."""
    s = _suffix(use_dcmtk)
    written: Dict[str, Dict] = {}

    # Per-image: dicom_raw_uint16, preproc_tensor, pred_risk_factors_per_key
    raw_dir = FIXTURES_DIR / f"dicom_raw_uint16{s}"
    pp_dir = FIXTURES_DIR / f"preproc_tensor{s}"
    rfk_dir = FIXTURES_DIR / f"pred_risk_factors_per_key{s}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pp_dir.mkdir(parents=True, exist_ok=True)
    rfk_dir.mkdir(parents=True, exist_ok=True)

    for (view, side), arr in captured["dicom_raw_uint16"].items():
        rel = f"dicom_raw_uint16{s}/{VIEW_STR[view]}_{SIDE_STR[side]}.npy"
        path = FIXTURES_DIR / rel
        np.save(path, arr)
        written[rel] = _file_info(path, arr)

    for (view, side), arr in captured["preproc_tensor"].items():
        rel = f"preproc_tensor{s}/{VIEW_STR[view]}_{SIDE_STR[side]}.npy"
        path = FIXTURES_DIR / rel
        np.save(path, arr)
        written[rel] = _file_info(path, arr)

    for key, arr in captured["pred_risk_factors_per_key"].items():
        rel = f"pred_risk_factors_per_key{s}/{key}.npy"
        path = FIXTURES_DIR / rel
        np.save(path, arr)
        written[rel] = _file_info(path, arr)

    # Single-tensor fixtures
    single = {
        f"image_encoder_out{s}.npy": captured["image_encoder_out"],
        f"image_hidden_in_pool{s}.npy": captured["image_hidden_in_pool"],
        f"pool_hidden{s}.npy": captured["pool_hidden"],
        f"risk_factor_vector{s}.npy": captured["risk_factor_vector"],
        f"xai_hidden{s}.npy": captured["xai_hidden"],
        f"raw_logit{s}.npy": captured["raw_logit"],
        f"raw_sigmoid{s}.npy": captured["raw_sigmoid"],
        f"calibrated{s}.npy": captured["calibrated"],
    }
    for rel, arr in single.items():
        path = FIXTURES_DIR / rel
        np.save(path, arr)
        written[rel] = _file_info(path, arr)

    # JSON: predictions + batch_order
    pred_obj = {"predictions": captured["predictions"], "modelVersion": captured["modelVersion"]}
    pred_path = FIXTURES_DIR / f"predictions{s}.json"
    pred_path.write_text(json.dumps(pred_obj, indent=2) + "\n")
    written[f"predictions{s}.json"] = {"sha256": _sha256(pred_path)}

    order_obj = [
        {
            "slot": i,
            "view": v,
            "view_str": VIEW_STR[v],
            "side": s_,
            "side_str": SIDE_STR[s_],
        }
        for i, (v, s_) in enumerate(captured["_batch_slot_to_view_side"])
    ]
    order_path = FIXTURES_DIR / f"batch_order{s}.json"
    order_path.write_text(json.dumps(order_obj, indent=2) + "\n")
    written[f"batch_order{s}.json"] = {"sha256": _sha256(order_path)}

    return written


def _file_info(path: pathlib.Path, arr: np.ndarray) -> Dict[str, Any]:
    return {
        "sha256": _sha256(path),
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
    }


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_previews(captured: Dict[str, Any]) -> None:
    """Render preproc_tensor (pydicom path) per-image to PNG for visual eyeballing.

    The preproc tensor has been mean/std-normalized. We undo that for display
    using the mean/std from the config.
    """
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    from onconet.predict import _load_config

    cfg = _load_config(str(MIRAI_CONFIG))
    mean = float(cfg.img_mean[0])
    std = float(cfg.img_std[0])

    for (view, side), arr in captured["preproc_tensor"].items():
        # arr shape: (3, H, W). Channel 0 is the original grayscale (replicated).
        gray = arr[0] * std + mean
        gray = np.clip(gray, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        # Downsample for previews so they aren't huge.
        h, w = gray.shape
        scale = max(1, max(h, w) // 800)
        if scale > 1:
            gray = gray[::scale, ::scale]
        img = Image.fromarray(gray, mode="I;16")
        # Convert to 8-bit for JPEG-safe display.
        img8 = (gray.astype(np.float32) / max(gray.max(), 1) * 255).astype(np.uint8)
        Image.fromarray(img8, mode="L").save(
            PREVIEW_DIR / f"{VIEW_STR[view]}_{SIDE_STR[side]}.png"
        )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def collect_env_info() -> Dict[str, Any]:
    import torchvision
    import pydicom
    import PIL

    def _git_sha(repo: pathlib.Path) -> Tuple[str, bool]:
        try:
            sha = subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            dirty = bool(subprocess.check_output(
                ["git", "-C", str(repo), "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
            ).decode().strip())
            return sha, dirty
        except Exception:
            return "", False

    def _dcmtk_version() -> str:
        try:
            out = subprocess.check_output(
                ["dcmj2pnm", "--version"], stderr=subprocess.STDOUT
            ).decode()
            for line in out.splitlines():
                stripped = line.strip()
                if stripped.startswith("$dcmtk:") or stripped.startswith("dcmtk:"):
                    return stripped
            for line in out.splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("W:") and not stripped.startswith("E:"):
                    return stripped
            return "unknown"
        except Exception:
            return "not installed"

    mirai_sha, mirai_dirty = _git_sha(MIRAI_ROOT)
    proj_sha, proj_dirty = _git_sha(PROJECT_ROOT)

    return {
        "platform": f"{platform.system()}-{platform.release()}-{platform.machine()}",
        "python_version": platform.python_version(),
        "python_arch": platform.machine(),
        "hostname": socket.gethostname(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "numpy_version": np.__version__,
        "pydicom_version": pydicom.__version__,
        "pillow_version": PIL.__version__,
        "dcmtk_version": _dcmtk_version(),
        "git": {
            "mirai_sha": mirai_sha,
            "mirai_dirty": mirai_dirty,
            "project_sha": proj_sha,
            "project_dirty": proj_dirty,
        },
    }


def collect_snapshot_info(config) -> Dict[str, Any]:
    paths = {
        "img_encoder_snapshot": config.img_encoder_snapshot,
        "transformer_snapshot": config.transformer_snapshot,
        "calibrator_path": config.calibrator_path,
    }
    info = {}
    for name, p in paths.items():
        path = pathlib.Path(os.path.expanduser(p))
        if path.exists():
            info[name] = {
                "path": p,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        else:
            info[name] = {"path": p, "sha256": None, "size_bytes": None}
    info["remote_snapshot_uri"] = getattr(config, "remote_snapshot_uri", None)
    return info


def collect_demo_data_info() -> Dict[str, Any]:
    return {
        "remote_uri": "https://github.com/reginabarzilaygroup/Mirai/releases/latest/download/mirai_demo_data.zip",
        "files": [
            {
                "name": name,
                "sha256": _sha256(DEMO_DATA_DIR / name),
                "size_bytes": (DEMO_DATA_DIR / name).stat().st_size,
            }
            for name in DEMO_FILES
        ],
    }


def write_manifest(
    runs: Dict[str, Dict[str, Any]],
    written_per_run: Dict[str, Dict[str, Dict]],
    determinism: Dict[str, Any],
) -> None:
    from onconet.predict import _load_config

    cfg = _load_config(str(MIRAI_CONFIG))

    # Aggregate the per-run shapes for the manifest. (Both runs should agree on
    # rf_dim, image_encoder_out shape, etc., but we record per-run anyway.)
    def shapes(c: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "rf_dim": c["rf_dim"],
            "preproc_batch_x": list(c["preproc_batch_x_shape"]),
            "image_encoder_out": list(c["image_encoder_out"].shape),
            "pool_hidden": list(c["pool_hidden"].shape),
            "risk_factor_vector": list(c["risk_factor_vector"].shape),
            "xai_hidden": list(c["xai_hidden"].shape),
            "raw_logit": list(c["raw_logit"].shape),
            "raw_sigmoid": list(c["raw_sigmoid"].shape),
            "calibrated": list(c["calibrated"].shape),
        }

    asserts_passed = [
        "all four (view,side) tuples present, no duplicates",
        "no NaN or Inf in any captured tensor",
        "xai_hidden ~= relu(pool_hidden)",
        "xai_hidden[:, -rf_dim:] ~= relu(risk_factor_vector)",
        "concat(per-key probs in risk_factor_keys order) ~= risk_factor_vector",
        "round(calibrated, 4) == predictions",
        "raw_logit monotone-nondecreasing across years",
    ]

    manifest = {
        "schema_version": 1,
        "captured_at_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "env": collect_env_info(),
        "config": {
            "config_path_from_mirai": str(MIRAI_CONFIG.relative_to(MIRAI_ROOT)),
            "config_sha256": _sha256(MIRAI_CONFIG),
            "img_size_w_h": list(cfg.img_size),
            "img_mean": list(cfg.img_mean),
            "img_std": list(cfg.img_std),
            "max_followup": cfg.max_followup,
            "use_pred_risk_factors_at_test": cfg.use_pred_risk_factors_at_test,
            "use_pred_risk_factors_if_unk": getattr(cfg, "use_pred_risk_factors_if_unk", False),
        },
        "snapshots": collect_snapshot_info(cfg),
        "demo_data": collect_demo_data_info(),
        "determinism": determinism,
        "asserts_passed": asserts_passed,
        "runs": {},
    }

    for run_name, captured in runs.items():
        manifest["runs"][run_name] = {
            "predictions": captured["predictions"],
            "modelVersion": captured["modelVersion"],
            "batch_order": [
                {
                    "slot": i,
                    "view": v,
                    "view_str": VIEW_STR[v],
                    "side": s_,
                    "side_str": SIDE_STR[s_],
                }
                for i, (v, s_) in enumerate(captured["_batch_slot_to_view_side"])
            ],
            "shapes": shapes(captured),
            "risk_factor_keys": captured["risk_factor_keys"],
            "fixture_files": written_per_run[run_name],
        }

    (FIXTURES_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _ensure_fresh(regenerate: bool, runs: List[str]) -> None:
    """If fixtures already exist, refuse unless regenerate=True."""
    sentinels = []
    for run in runs:
        s = "_dcmtk" if run == "dcmtk" else ""
        sentinels.append(FIXTURES_DIR / f"raw_logit{s}.npy")
    existing = [p for p in sentinels if p.exists()]
    if existing and not regenerate:
        names = ", ".join(p.name for p in existing)
        raise SystemExit(
            f"Refusing to overwrite existing fixtures: {names}. "
            f"Pass --regenerate to overwrite."
        )
    if regenerate:
        # Wipe per-run dirs so stale files don't linger.
        for run in runs:
            s = "_dcmtk" if run == "dcmtk" else ""
            for sub in (
                f"dicom_raw_uint16{s}",
                f"preproc_tensor{s}",
                f"pred_risk_factors_per_key{s}",
            ):
                d = FIXTURES_DIR / sub
                if d.exists():
                    shutil.rmtree(d)


def _verify_demo_data() -> None:
    missing = [n for n in DEMO_FILES if not (DEMO_DATA_DIR / n).exists()]
    if missing:
        raise SystemExit(
            f"Demo DICOMs missing under {DEMO_DATA_DIR}: {missing}. "
            f"Download via: curl -sLO https://github.com/reginabarzilaygroup/Mirai/"
            f"releases/latest/download/mirai_demo_data.zip && "
            f"unzip -o mirai_demo_data.zip -d mirai_demo_data"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate", action="store_true", help="Overwrite existing fixtures.")
    parser.add_argument(
        "--only",
        choices=["pydicom", "dcmtk", "both"],
        default="both",
        help="Restrict to one DICOM-decoding path (default: both).",
    )
    args = parser.parse_args()

    runs = {
        "both": ["pydicom", "dcmtk"],
        "pydicom": ["pydicom"],
        "dcmtk": ["dcmtk"],
    }[args.only]

    determinism = set_determinism()
    _verify_demo_data()
    _ensure_fresh(args.regenerate, runs)

    captured_per_run: Dict[str, Dict[str, Any]] = {}
    written_per_run: Dict[str, Dict[str, Dict]] = {}

    for run in runs:
        use_dcmtk = run == "dcmtk"
        print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] Capturing {run}...", flush=True)
        captured = run_pipeline_with_hooks(use_dcmtk=use_dcmtk)
        captured_per_run[run] = captured
        written_per_run[run] = write_fixtures(captured, use_dcmtk=use_dcmtk)
        print(
            f"[{dt.datetime.now().isoformat(timespec='seconds')}] {run} predictions: {captured['predictions']}",
            flush=True,
        )

    if "pydicom" in captured_per_run:
        write_previews(captured_per_run["pydicom"])

    write_manifest(captured_per_run, written_per_run, determinism)
    print(f"Wrote manifest with {sum(len(w) for w in written_per_run.values())} fixture entries.")


if __name__ == "__main__":
    main()
