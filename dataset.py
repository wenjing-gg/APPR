import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
import torch
from monai.data import DataLoader, Dataset
import monai.transforms.compose as monai_compose_mod
import monai.transforms.transform as monai_transform_mod
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    ScaleIntensityRangePercentilesd,
    SpatialPadd,
)
import monai.utils.misc as monai_misc_mod


MONAI_SAFE_MAX_SEED = int(np.iinfo(np.uint32).max)
DEFAULT_DATASET_ROOT = "/data/cyf/codes/tooth/NasalSeg"
ORIENTATION_TOL = 1e-4


@dataclass
class CaseItem:
    case_id: str
    image: str
    label: Optional[str] = None


def patch_monai_max_seed() -> None:
    monai_compose_mod.MAX_SEED = MONAI_SAFE_MAX_SEED
    monai_transform_mod.MAX_SEED = MONAI_SAFE_MAX_SEED
    monai_misc_mod.MAX_SEED = MONAI_SAFE_MAX_SEED


def strip_ext(name: str) -> str:
    n = os.path.basename(name)
    if n.endswith(".nii.gz"):
        return n[:-7]
    if n.endswith(".nii"):
        return n[:-4]
    return os.path.splitext(n)[0]


def ensure_exists(path: str, what: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {what}: {path}")
    return path


def _first_existing(candidates: Sequence[str]) -> Optional[str]:
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def resolve_case_paths(dataset_root: str, case_name: str, require_label: bool = True) -> Tuple[str, Optional[str]]:
    case_id = strip_ext(case_name)
    case_name = os.path.basename(case_name)
    root = os.path.abspath(dataset_root)

    image_candidates = [
        os.path.join(root, "imagesTr", f"{case_id}_0000.nii.gz"),
        os.path.join(root, "imagesTr", f"{case_id}_0000.nii"),
        os.path.join(root, "imagesTr", f"{case_id}.nii.gz"),
        os.path.join(root, "imagesTr", f"{case_id}.nii"),
        os.path.join(root, "imagesTr", case_name),
        os.path.join(root, "images", f"{case_id}_img.nrrd"),
        os.path.join(root, "images", f"{case_id}.nrrd"),
        os.path.join(root, "images", f"{case_id}_img.nii.gz"),
        os.path.join(root, "images", f"{case_id}_img.nii"),
        os.path.join(root, "images", f"{case_id}.nii.gz"),
        os.path.join(root, "images", f"{case_id}.nii"),
        os.path.join(root, "images", case_name),
    ]
    label_candidates = [
        os.path.join(root, "labels_sinus_fixed", f"{case_id}_seg.nrrd"),
        os.path.join(root, "labels_sinus_fixed", f"{case_id}.nrrd"),
        os.path.join(root, "labels_sinus_fixed", f"{case_id}_seg.nii.gz"),
        os.path.join(root, "labels_sinus_fixed", f"{case_id}_seg.nii"),
        os.path.join(root, "labels_sinus_fixed", case_name),
        os.path.join(root, "labels_sinus", f"{case_id}_seg.nrrd"),
        os.path.join(root, "labels_sinus", f"{case_id}.nrrd"),
        os.path.join(root, "labels_sinus", f"{case_id}_seg.nii.gz"),
        os.path.join(root, "labels_sinus", f"{case_id}_seg.nii"),
        os.path.join(root, "labels_sinus", case_name),
    ]

    image_path = _first_existing(list(dict.fromkeys(image_candidates)))
    label_path = _first_existing(list(dict.fromkeys(label_candidates)))

    if image_path is None:
        raise FileNotFoundError(f"Image not found for case {case_id}: tried {image_candidates}")
    if require_label and label_path is None:
        raise FileNotFoundError(f"Label not found for case {case_id}: tried {label_candidates}")
    return image_path, label_path


def load_split_cases(dataset_root: str, split_json: str, split_ratio: str) -> Tuple[List[CaseItem], List[CaseItem]]:
    ensure_exists(split_json, "split json")
    with open(split_json, "r", encoding="utf-8") as file_obj:
        splits = json.load(file_obj)

    train_splits = splits.get("train", {})
    if split_ratio not in train_splits:
        raise KeyError(f"split ratio {split_ratio!r} not found. Available: {list(train_splits.keys())}")

    labeled_cases = train_splits[split_ratio].get("labeled", [])
    unlabeled_cases = train_splits[split_ratio].get("unlabeled", [])

    labeled_items: List[CaseItem] = []
    unlabeled_items: List[CaseItem] = []

    for name in labeled_cases:
        case_id = strip_ext(name)
        image_path, label_path = resolve_case_paths(dataset_root, name, require_label=True)
        labeled_items.append(CaseItem(case_id=case_id, image=image_path, label=label_path))

    for name in unlabeled_cases:
        case_id = strip_ext(name)
        image_path, _ = resolve_case_paths(dataset_root, name, require_label=False)
        unlabeled_items.append(CaseItem(case_id=case_id, image=image_path, label=None))

    return labeled_items, unlabeled_items


def load_validation_cases(dataset_root: str, split_json: str, split_ratio: str, val_key: str) -> List[CaseItem]:
    key = str(val_key).strip()
    if key == "":
        return []

    ensure_exists(split_json, "split json")
    with open(split_json, "r", encoding="utf-8") as file_obj:
        splits = json.load(file_obj)

    case_names: Optional[List[str]] = None
    if isinstance(splits.get(key, None), list):
        case_names = splits[key]
    else:
        train_splits = splits.get("train", {})
        ratio_splits = train_splits.get(split_ratio, {}) if isinstance(train_splits, dict) else {}
        if isinstance(ratio_splits, dict) and isinstance(ratio_splits.get(key, None), list):
            case_names = ratio_splits[key]

    if case_names is None:
        available_top = [k for k, value in splits.items() if isinstance(value, list)]
        train_keys = []
        train_splits = splits.get("train", {})
        if isinstance(train_splits, dict) and split_ratio in train_splits and isinstance(train_splits[split_ratio], dict):
            train_keys = list(train_splits[split_ratio].keys())
        raise KeyError(
            f"val key {key!r} not found in split json. top-level list keys={available_top}, "
            f"train[{split_ratio}] keys={train_keys}"
        )

    val_items: List[CaseItem] = []
    for name in case_names:
        case_id = strip_ext(name)
        image_path, label_path = resolve_case_paths(dataset_root, name, require_label=True)
        val_items.append(CaseItem(case_id=case_id, image=image_path, label=label_path))
    return val_items


def _read_image_direction(path: str) -> np.ndarray:
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.ReadImageInformation()
    direction = np.asarray(reader.GetDirection(), dtype=np.float64)
    if direction.size != 9:
        raise ValueError(f"Expected a 3D image direction for {path}, got {tuple(direction.shape)} with {direction.size} values.")
    return direction.reshape(3, 3)


def _array_axis_to_image_axis(array_axis: int) -> int:
    if array_axis not in (0, 1, 2):
        raise ValueError(f"Expected array axis in [0, 1, 2], got {array_axis}.")
    # sitk image axes are (x, y, z), while numpy arrays are (z, y, x)
    return 2 - int(array_axis)


def summarize_orientation_consistency(
    cases: Sequence[CaseItem],
    lateral_axis: int = -1,
    atol: float = ORIENTATION_TOL,
) -> Dict[str, object]:
    if len(cases) == 0:
        return {
            "num_cases": 0,
            "direction": tuple(),
            "array_axis": int(lateral_axis),
            "image_axis": None,
            "increasing_toward": "unknown",
            "label_header_mismatch_count": 0,
            "label_header_mismatch_cases": [],
        }

    array_axis = int(lateral_axis) % 3
    image_axis = _array_axis_to_image_axis(array_axis)
    ref_case = cases[0]
    ref_dir = _read_image_direction(ref_case.image)

    mismatched_cases: List[str] = []
    label_mismatches: List[str] = []
    for case in cases:
        img_dir = _read_image_direction(case.image)
        if not np.allclose(img_dir, ref_dir, atol=float(atol), rtol=0.0):
            mismatched_cases.append(case.case_id)
        if case.label:
            lbl_dir = _read_image_direction(case.label)
            if not np.allclose(lbl_dir, img_dir, atol=float(atol), rtol=0.0):
                label_mismatches.append(case.case_id)

    if mismatched_cases:
        ref_flat = tuple(round(float(x), 6) for x in ref_dir.reshape(-1))
        raise ValueError(
            "Inconsistent image directions detected. "
            f"Reference case={ref_case.case_id}, direction={ref_flat}, mismatched_cases={mismatched_cases[:8]}"
        )

    lateral_vec = ref_dir[:, image_axis]
    dominant_physical_axis = int(np.argmax(np.abs(lateral_vec)))
    dominant_value = float(lateral_vec[dominant_physical_axis])
    if dominant_physical_axis != 0:
        raise ValueError(
            "The configured lateral axis does not map to the physical left-right axis. "
            f"array_axis={array_axis}, image_axis={image_axis}, direction_column={tuple(round(float(x), 6) for x in lateral_vec)}"
        )
    if abs(abs(dominant_value) - 1.0) > 5e-3:
        raise ValueError(
            "The configured lateral axis is oblique relative to the physical left-right axis. "
            f"array_axis={array_axis}, image_axis={image_axis}, direction_column={tuple(round(float(x), 6) for x in lateral_vec)}"
        )

    increasing_toward = "left" if dominant_value > 0 else "right"
    return {
        "num_cases": int(len(cases)),
        "direction": tuple(round(float(x), 6) for x in ref_dir.reshape(-1)),
        "array_axis": int(array_axis),
        "image_axis": int(image_axis),
        "increasing_toward": increasing_toward,
        "label_header_mismatch_count": int(len(label_mismatches)),
        "label_header_mismatch_cases": label_mismatches,
    }


def build_train_transform(roi_size: Sequence[int], p_low: float, p_high: float, num_samples: int) -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRangePercentilesd(
                keys="image",
                lower=float(p_low),
                upper=float(p_high),
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=tuple(int(x) for x in roi_size),
                pos=2,
                neg=1,
                num_samples=max(1, int(num_samples)),
                allow_smaller=True,
            ),
            SpatialPadd(keys=["image", "label"], spatial_size=tuple(int(x) for x in roi_size)),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0]),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[1]),
            # Do not flip along the lateral axis: cls1/cls2 encode fixed side identity.
            RandRotate90d(keys=["image", "label"], prob=0.25, max_k=3),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def build_array_train_transform(roi_size: Sequence[int], num_samples: int) -> Compose:
    return Compose(
        [
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=tuple(int(x) for x in roi_size),
                pos=2,
                neg=1,
                num_samples=max(1, int(num_samples)),
                allow_smaller=True,
            ),
            SpatialPadd(keys=["image", "label"], spatial_size=tuple(int(x) for x in roi_size)),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0]),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[1]),
            # Keep pseudo labels on a fixed left/right convention during augmentation as well.
            RandRotate90d(keys=["image", "label"], prob=0.25, max_k=3),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def build_array_crop_transform(roi_size: Sequence[int], num_samples: int) -> Compose:
    return Compose(
        [
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=tuple(int(x) for x in roi_size),
                pos=2,
                neg=1,
                num_samples=max(1, int(num_samples)),
                allow_smaller=True,
            ),
            SpatialPadd(keys=["image", "label"], spatial_size=tuple(int(x) for x in roi_size)),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def build_array_guided_crop_transform(
    roi_size: Sequence[int],
    num_samples: int,
    prior_mask_key: str = "prior_mask",
    extra_keys: Optional[Sequence[str]] = None,
) -> Compose:
    keys = ["image", str(prior_mask_key)]
    if extra_keys:
        for key in extra_keys:
            key_str = str(key)
            if key_str not in keys:
                keys.append(key_str)

    return Compose(
        [
            EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
            RandCropByPosNegLabeld(
                keys=keys,
                label_key=str(prior_mask_key),
                spatial_size=tuple(int(x) for x in roi_size),
                pos=2,
                neg=1,
                num_samples=max(1, int(num_samples)),
                allow_smaller=True,
            ),
            SpatialPadd(keys=keys, spatial_size=tuple(int(x) for x in roi_size)),
            EnsureTyped(keys=keys),
        ]
    )


def build_array_spatial_transform() -> Compose:
    return Compose(
        [
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0]),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[1]),
            # Keep pseudo labels on a fixed left/right convention during augmentation as well.
            RandRotate90d(keys=["image", "label"], prob=0.25, max_k=3),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def build_loader(
    data: List[Dict[str, str]],
    transform: Compose,
    batch_size: int,
    workers: int,
    drop_last: bool = False,
) -> Optional[DataLoader]:
    if len(data) == 0:
        return None
    dataset = Dataset(data=data, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=drop_last,
    )


def endless_iter(loader: DataLoader) -> Iterable[Dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def normalize_intensity_np(vol: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    x = np.asarray(vol, dtype=np.float32)
    lo = float(np.percentile(x, p_low))
    hi = float(np.percentile(x, p_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo)
    return x.astype(np.float32)
