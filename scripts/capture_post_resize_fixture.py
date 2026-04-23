"""Phase 6 helper: capture the intermediate int32 tensor produced by
PIL's bilinear resize (the `Scale_2d` transform) when fed the Phase 0
`dicom_raw_uint16` arrays. This isolates the resize step so the TS
port of PIL bilinear can be tested in isolation from windowing + align.

Produces `tests/reference/fixtures/post_resize/{CC,MLO}_{L,R}.npy`
(int32, shape (2048, 1664)). Also produces `post_resize_dcmtk/*.npy`
for completeness.

Run under `mirai-py38` so Pillow/torchvision versions match Phase 0:

    conda activate mirai-py38
    python scripts/capture_post_resize_fixture.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch  # noqa: F401  (pinned torch 1.9.0 — needed by torchvision)
import torchvision
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "reference" / "fixtures"
MANIFEST_PATH = FIXTURES / "MANIFEST.json"

IMG_SIZE_W_H = (1664, 2048)  # torchvision.Scale_2d unpacks img_size as (width, height)
SIDES = ["CC_L", "CC_R", "MLO_L", "MLO_R"]


def _pil_bilinear_resize_mode_I(u16: np.ndarray) -> np.ndarray:
    """Replicate the exact path `Scale_2d` takes: uint16 → int32 → PIL mode 'I' →
    torchvision.transforms.Resize((H=2048, W=1664)) default-bilinear → int32 ndarray.
    """
    assert u16.dtype == np.uint16
    i32 = u16.astype(np.int32)
    pil = Image.fromarray(i32, mode="I")
    transform = torchvision.transforms.Resize((IMG_SIZE_W_H[1], IMG_SIZE_W_H[0]))  # (H, W)
    out_pil = transform(pil)
    out = np.asarray(out_pil, dtype=np.int32).copy()
    assert out.shape == (IMG_SIZE_W_H[1], IMG_SIZE_W_H[0]), out.shape
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _capture(variant: str) -> dict[str, dict[str, object]]:
    """variant: '' for pydicom; '_dcmtk' for dcmtk."""
    src_dir = FIXTURES / f"dicom_raw_uint16{variant}"
    dst_dir = FIXTURES / f"post_resize{variant}"
    dst_dir.mkdir(exist_ok=True)

    manifest_entries: dict[str, dict[str, object]] = {}
    for label in SIDES:
        src = src_dir / f"{label}.npy"
        dst = dst_dir / f"{label}.npy"
        u16 = np.load(src)
        resized = _pil_bilinear_resize_mode_I(u16)
        np.save(dst, resized, allow_pickle=False)
        sha = _sha256(dst)
        rel = f"post_resize{variant}/{label}.npy"
        manifest_entries[rel] = {
            "sha256": sha,
            "dtype": str(resized.dtype),
            "shape": list(resized.shape),
        }
        print(f"  {rel}: sha={sha[:12]}… shape={resized.shape} min={resized.min()} max={resized.max()} mean={resized.mean():.2f}")
    return manifest_entries


def _update_manifest(
    pydicom_entries: dict[str, dict[str, object]],
    dcmtk_entries: dict[str, dict[str, object]],
) -> None:
    with MANIFEST_PATH.open() as f:
        manifest = json.load(f)

    for rel, meta in pydicom_entries.items():
        manifest["runs"]["pydicom"]["fixture_files"][rel] = meta
    for rel, meta in dcmtk_entries.items():
        manifest["runs"]["dcmtk"]["fixture_files"][rel] = meta

    with MANIFEST_PATH.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nUpdated {MANIFEST_PATH.relative_to(REPO)} with {len(pydicom_entries) + len(dcmtk_entries)} entries")


def main() -> int:
    print("== pydicom path ==")
    pydicom_entries = _capture("")
    print("\n== dcmtk path ==")
    dcmtk_dir = FIXTURES / "dicom_raw_uint16_dcmtk"
    if dcmtk_dir.exists():
        dcmtk_entries = _capture("_dcmtk")
    else:
        print("  skipped — dicom_raw_uint16_dcmtk/ absent")
        dcmtk_entries = {}

    _update_manifest(pydicom_entries, dcmtk_entries)
    print(f"Pillow version: {Image.__version__}")
    print(f"torchvision version: {torchvision.__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
