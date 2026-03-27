#!/usr/bin/env python3
"""PriorNet Otsu (refactored)

A self-contained maxillary sinus segmentation pipeline:
1) Build a robust head-inner mask.
2) Estimate an air reference threshold from inside-head intensities.
3) Produce an air probability map.
4) Threshold + bilateral connected-component selection for sinus segmentation.

This file intentionally avoids dependency on external `priornet.py`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk

try:
    from skimage.filters import threshold_otsu as skimage_threshold_otsu
except Exception:  # pragma: no cover - optional dependency
    skimage_threshold_otsu = None


# -----------------------------
# Configuration
# -----------------------------


@dataclass
class PriorNetConfig:
    # Head mask construction
    tissue_hu_threshold: float = -300.0
    head_closing_mm: float = 7.0

    # Air reference estimation
    air_quantile: float = 0.08
    air_threshold_min_hu: float = -1024.0
    air_threshold_max_hu: float = -350.0
    use_otsu_air_cut: bool = True
    otsu_blend_weight: float = 1.0
    otsu_clip_min_hu: float = -1024.0
    otsu_clip_max_hu: float = -450.0
    otsu_min_voxels: int = 1024
    otsu_sample_max_voxels: int = 2000000
    air_cut_min_hu: float = -900.0
    air_cut_max_hu: float = -550.0
    probability_reference_threshold: float = 0.10

    # Probability shaping
    probability_sharpness_hu: float = 120.0
    preprocess_method: str = "gaussian"  # gaussian, median, none
    gaussian_sigma_mm: float = 0.8
    median_radius_vox: int = 1
    use_head_roi_gate: bool = False
    otsu_on_global_image: bool = True

    # Geometry prior (array order: z, y, x)
    roi_z_min: float = 0.12
    roi_z_max: float = 0.86
    roi_y_min: float = 0.18
    roi_y_max: float = 0.70
    exclude_midline: bool = True
    midline_exclude_ratio: float = 0.04

    # Component filtering
    min_component_voxels: int = 500
    target_y_rel: float = 0.42
    target_z_rel: float = 0.50
    target_x_rel_left: float = 0.22
    target_x_rel_right: float = 0.78
    sigma_y_rel: float = 0.16
    sigma_z_rel: float = 0.20
    sigma_x_rel: float = 0.14
    component_loc_weight: float = 0.65
    side_min_voxels: int = 64
    side_balance_min_ratio: float = 0.03
    side_balance_target_ratio: float = 0.20
    side_balance_weight: float = 0.35
    enforce_bilateral: bool = True
    max_candidates_per_side: int = 8
    max_components_per_side: int = 1
    enable_candidate_location_filter: bool = True
    candidate_y_min_rel: float = 0.30
    candidate_y_max_rel: float = 0.60
    candidate_z_min_rel: float = 0.30
    candidate_z_max_rel: float = 0.65
    candidate_left_x_max_rel: float = 0.45
    candidate_right_x_min_rel: float = 0.55
    split_components_by_side: bool = True
    pre_component_opening_mm: float = 0.0
    pre_component_midline_cut_vox: int = 0
    remove_border_connected_air: bool = True
    opening_mm: float = 0.8
    closing_mm: float = 1.2
    fill_holes_after_morph: bool = True


# -----------------------------
# Utilities
# -----------------------------


def sigmoid(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def dice_coefficient(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = int(np.logical_and(pred_b, gt_b).sum())
    den = int(pred_b.sum() + gt_b.sum())
    return float((2.0 * inter) / max(den, 1))


def iou_coefficient(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = int(np.logical_and(pred_b, gt_b).sum())
    union = int(np.logical_or(pred_b, gt_b).sum())
    return float(inter / max(union, 1))


def build_thresholds(start: float, end: float, step: float) -> np.ndarray:
    step = max(float(step), 1e-6)
    n = int(np.floor((float(end) - float(start)) / step)) + 1
    n = max(n, 1)
    vals = np.asarray([float(start) + i * step for i in range(n)], dtype=np.float32)
    vals = np.clip(vals, 0.0, 1.0)
    return np.unique(vals)


def metrics_sort_key(row: dict, metric_key: str) -> Tuple[float, float, float, float]:
    return (
        float(row.get(metric_key, float("-inf"))),
        float(row.get("dice_balanced", float("-inf"))),
        float(row.get("dice", float("-inf"))),
        float(row.get("iou", float("-inf")),),
    )


# -----------------------------
# PriorNet class
# -----------------------------


class PriorNet:
    def __init__(self, cfg: Optional[PriorNetConfig] = None):
        self.cfg = cfg or PriorNetConfig()

    # IO
    def load_volume(self, path: str) -> Tuple[np.ndarray, dict]:
        image = sitk.ReadImage(path)
        vol = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
        meta = {"sitk_image": image, "type": "sitk"}
        return vol, meta

    def save_volume(self, path: str, vol: np.ndarray, meta: dict) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        img = sitk.GetImageFromArray(vol)
        img.CopyInformation(meta["sitk_image"])
        sitk.WriteImage(img, path, useCompression=True)

    # Core pipeline
    def _head_mask(self, vol: np.ndarray, meta: dict) -> np.ndarray:
        base_img = meta["sitk_image"]
        tissue = sitk.Cast(base_img > float(self.cfg.tissue_hu_threshold), sitk.sitkUInt8)

        cc = sitk.ConnectedComponent(tissue)
        cc = sitk.RelabelComponent(cc, sortByObjectSize=True)
        head = sitk.Equal(cc, 1)

        sx, sy, sz = base_img.GetSpacing()
        radius = [
            max(1, int(round(float(self.cfg.head_closing_mm) / max(float(sx), 1e-6)))),
            max(1, int(round(float(self.cfg.head_closing_mm) / max(float(sy), 1e-6)))),
            max(1, int(round(float(self.cfg.head_closing_mm) / max(float(sz), 1e-6)))),
        ]
        head = sitk.BinaryMorphologicalClosing(head, radius)
        head = sitk.BinaryFillhole(head)
        return sitk.GetArrayFromImage(head).astype(bool, copy=False)

    @staticmethod
    def _radius_from_mm(ref_image: sitk.Image, radius_mm: float) -> List[int]:
        if radius_mm <= 0:
            return [0, 0, 0]
        sx, sy, sz = ref_image.GetSpacing()
        return [
            max(1, int(round(float(radius_mm) / max(float(sx), 1e-6)))),
            max(1, int(round(float(radius_mm) / max(float(sy), 1e-6)))),
            max(1, int(round(float(radius_mm) / max(float(sz), 1e-6)))),
        ]

    @staticmethod
    def _radius_from_mm_allow_zero(ref_image: sitk.Image, radius_mm: float) -> List[int]:
        if radius_mm <= 0:
            return [0, 0, 0]
        sx, sy, sz = ref_image.GetSpacing()
        return [
            max(0, int(round(float(radius_mm) / max(float(sx), 1e-6)))),
            max(0, int(round(float(radius_mm) / max(float(sy), 1e-6)))),
            max(0, int(round(float(radius_mm) / max(float(sz), 1e-6)))),
        ]

    @staticmethod
    def _remove_border_touching_components(binary_mask: np.ndarray) -> np.ndarray:
        # Remove only the largest border-touching component as external-air surrogate.
        # If every component touches border, keep all to avoid collapsing valid cavities.
        if binary_mask.size == 0:
            return binary_mask

        cc = sitk.ConnectedComponent(sitk.GetImageFromArray(binary_mask.astype(np.uint8)))
        cc_arr = sitk.GetArrayFromImage(cc)
        if cc_arr.max() <= 0:
            return binary_mask

        border = np.zeros_like(binary_mask, dtype=bool)
        border[0, :, :] = True
        border[-1, :, :] = True
        border[:, 0, :] = True
        border[:, -1, :] = True
        border[:, :, 0] = True
        border[:, :, -1] = True

        border_labels = np.unique(cc_arr[border])
        border_labels = border_labels[border_labels > 0]
        if border_labels.size == 0:
            return binary_mask

        all_labels = np.unique(cc_arr)
        all_labels = all_labels[all_labels > 0]
        border_set = set(int(x) for x in border_labels.tolist())
        if len(border_set) >= len(all_labels):
            # Nothing clearly internal by connectivity; keep to avoid deleting everything.
            return binary_mask

        largest_label = None
        largest_size = -1
        for lb in border_labels:
            sz = int((cc_arr == int(lb)).sum())
            if sz > largest_size:
                largest_size = sz
                largest_label = int(lb)

        if largest_label is None:
            return binary_mask
        keep = binary_mask.copy()
        keep[cc_arr == int(largest_label)] = False
        return keep

    def _preprocess_volume(self, vol: np.ndarray, meta: dict) -> np.ndarray:
        method = str(self.cfg.preprocess_method).strip().lower()
        if method in ("none", "", "off"):
            return vol.astype(np.float32, copy=False)

        base_img = meta["sitk_image"]
        if method == "gaussian":
            sigma = float(max(self.cfg.gaussian_sigma_mm, 0.0))
            if sigma <= 0:
                return vol.astype(np.float32, copy=False)
            out_img = sitk.SmoothingRecursiveGaussian(base_img, sigma=sigma)
            return sitk.GetArrayFromImage(out_img).astype(np.float32, copy=False)

        if method == "median":
            r = int(max(0, self.cfg.median_radius_vox))
            if r <= 0:
                return vol.astype(np.float32, copy=False)
            out_img = sitk.Median(base_img, [r, r, r])
            return sitk.GetArrayFromImage(out_img).astype(np.float32, copy=False)

        raise ValueError(f"Unsupported preprocess_method: {self.cfg.preprocess_method}")

    def _air_threshold(self, vol: np.ndarray, head_mask: np.ndarray) -> float:
        # Keep legacy private method name for compatibility with previous scripts.
        return self._quantile_air_threshold(vol, head_mask)

    def _quantile_air_threshold(self, vol: np.ndarray, head_mask: np.ndarray) -> float:
        inside = vol[head_mask]
        if inside.size == 0:
            inside = vol.ravel()
        if inside.size == 0:
            return float(self.cfg.air_threshold_min_hu)

        q = float(np.clip(self.cfg.air_quantile, 0.0, 1.0))
        thr = float(np.quantile(inside, q))
        return float(np.clip(thr, self.cfg.air_threshold_min_hu, self.cfg.air_threshold_max_hu))

    def _otsu_air_cut_hu(self, values: np.ndarray) -> Tuple[float, int]:
        if not bool(self.cfg.use_otsu_air_cut):
            raise RuntimeError("Otsu air-cut is disabled by config (`use_otsu_air_cut=False`).")
        if skimage_threshold_otsu is None:
            raise RuntimeError("Otsu air-cut requires scikit-image, but `threshold_otsu` is unavailable.")

        work = values
        clip_min = float(self.cfg.otsu_clip_min_hu)
        clip_max = float(self.cfg.otsu_clip_max_hu)
        if clip_max > clip_min:
            work = work[(work >= clip_min) & (work <= clip_max)]

        if work.size < int(max(1, self.cfg.otsu_min_voxels)):
            work = values
        if work.size < int(max(1, self.cfg.otsu_min_voxels)):
            raise RuntimeError(
                f"Insufficient voxels for Otsu air-cut: {int(work.size)} < {int(max(1, self.cfg.otsu_min_voxels))}."
            )

        max_sample = int(max(1, self.cfg.otsu_sample_max_voxels))
        if work.size > max_sample:
            rng = np.random.default_rng(0)
            idx = rng.choice(work.size, size=max_sample, replace=False)
            work = work[idx]

        try:
            otsu_hu = float(skimage_threshold_otsu(work.astype(np.float32)))
            return otsu_hu, int(work.size)
        except Exception:
            raise RuntimeError("Otsu air-cut failed on candidate voxels.")

    def _estimate_air_cut_hu(self, vol: np.ndarray, head_mask: np.ndarray, roi_mask: np.ndarray) -> Tuple[float, Dict[str, float]]:
        quant_hu = self._quantile_air_threshold(vol, head_mask)

        if bool(self.cfg.otsu_on_global_image):
            # Strict Otsu workflow: compute threshold on (preprocessed) whole image histogram.
            otsu_values = vol.ravel()
        else:
            # Optional constrained variant.
            roi_head = head_mask & roi_mask
            otsu_values = vol[roi_head]
            if otsu_values.size == 0:
                otsu_values = vol[head_mask]
        if otsu_values.size == 0:
            otsu_values = vol.ravel()

        otsu_hu, otsu_count = self._otsu_air_cut_hu(otsu_values)
        w_otsu = float(np.clip(self.cfg.otsu_blend_weight, 0.0, 1.0))
        air_cut = float((1.0 - w_otsu) * quant_hu + w_otsu * otsu_hu)

        cut_lo = float(min(self.cfg.air_cut_min_hu, self.cfg.air_cut_max_hu))
        cut_hi = float(max(self.cfg.air_cut_min_hu, self.cfg.air_cut_max_hu))
        air_cut = float(np.clip(air_cut, cut_lo, cut_hi))

        debug = {
            "air_threshold_quantile_hu": float(quant_hu),
            "air_threshold_otsu_hu": float(otsu_hu),
            "air_threshold_otsu_voxels": float(otsu_count),
            "air_cut_hu": float(air_cut),
        }
        return air_cut, debug

    def _geometry_roi(self, shape: Tuple[int, int, int], keep_midline: bool) -> np.ndarray:
        z_dim, y_dim, x_dim = [max(int(v), 1) for v in shape]

        z = np.arange(z_dim, dtype=np.float32).reshape(z_dim, 1, 1) / max(z_dim - 1, 1)
        y = np.arange(y_dim, dtype=np.float32).reshape(1, y_dim, 1) / max(y_dim - 1, 1)

        roi = (
            (z >= float(self.cfg.roi_z_min))
            & (z <= float(self.cfg.roi_z_max))
            & (y >= float(self.cfg.roi_y_min))
            & (y <= float(self.cfg.roi_y_max))
        )

        if self.cfg.exclude_midline and (not keep_midline):
            center_x = (x_dim - 1) * 0.5
            half_band = max(1, int(round(x_dim * float(self.cfg.midline_exclude_ratio) * 0.5)))
            x0 = max(0, int(center_x - half_band))
            x1 = min(x_dim, int(center_x + half_band + 1))
            roi = roi.copy()
            roi[:, :, x0:x1] = False

        return roi

    def compute_probability_map(self, vol: np.ndarray, meta: dict, keep_midline: bool = False) -> Tuple[np.ndarray, Dict[str, float]]:
        pre_vol = self._preprocess_volume(vol, meta)
        head_mask = self._head_mask(pre_vol, meta)
        roi = self._geometry_roi(pre_vol.shape, keep_midline=keep_midline)
        air_cut_hu, air_dbg = self._estimate_air_cut_hu(pre_vol, head_mask, roi)

        if bool(self.cfg.use_head_roi_gate):
            gate = (head_mask & roi).astype(np.float32)
        else:
            gate = np.ones(pre_vol.shape, dtype=np.float32)
        scale = max(float(self.cfg.probability_sharpness_hu), 1e-3)

        # Align HU cut with the external probability threshold convention used by callers.
        ref_prob = float(np.clip(self.cfg.probability_reference_threshold, 1e-4, 1.0 - 1e-4))
        center_hu = float(air_cut_hu + scale * np.log(ref_prob / (1.0 - ref_prob)))

        raw_prob = sigmoid((center_hu - pre_vol) / scale).astype(np.float32)
        prob = (raw_prob * gate).astype(np.float32)

        debug = {
            "air_threshold_hu": float(center_hu),
            "air_probability_center_hu": float(center_hu),
            "air_probability_ref_threshold": float(ref_prob),
            "preprocess_method": str(self.cfg.preprocess_method),
            "head_voxels": float(head_mask.sum()),
            "roi_voxels": float((head_mask & roi).sum()) if bool(self.cfg.use_head_roi_gate) else float(np.prod(pre_vol.shape)),
        }
        debug.update(air_dbg)
        return prob, debug

    def _component_candidates(
        self,
        binary_mask: np.ndarray,
        ref_image: sitk.Image,
    ) -> List[Tuple[float, int, str, int]]:
        mask_img = sitk.GetImageFromArray(binary_mask.astype(np.uint8))
        mask_img.CopyInformation(ref_image)

        cc = sitk.ConnectedComponent(mask_img)
        stat = sitk.LabelShapeStatisticsImageFilter()
        stat.Execute(cc)

        size_x, size_y, size_z = ref_image.GetSize()
        x_mid = (size_x - 1) * 0.5

        entries: List[Tuple[float, int, str, int]] = []
        for label_id in stat.GetLabels():
            voxels = int(stat.GetNumberOfPixels(label_id))
            if voxels < int(self.cfg.min_component_voxels):
                continue

            cx, cy, cz = stat.GetCentroid(label_id)
            ix, iy, iz = ref_image.TransformPhysicalPointToIndex((cx, cy, cz))

            side = "left" if ix < x_mid else "right"
            y_rel = float(iy / max(size_y - 1, 1))
            z_rel = float(iz / max(size_z - 1, 1))

            sy = np.exp(-0.5 * ((y_rel - float(self.cfg.target_y_rel)) / max(float(self.cfg.sigma_y_rel), 1e-6)) ** 2)
            sz = np.exp(-0.5 * ((z_rel - float(self.cfg.target_z_rel)) / max(float(self.cfg.sigma_z_rel), 1e-6)) ** 2)
            loc_score = float(sy * sz)

            w = float(np.clip(self.cfg.component_loc_weight, 0.0, 1.0))
            score = float(voxels) * ((1.0 - w) + w * loc_score)
            entries.append((score, int(label_id), side, voxels))

        return entries

    def segment_from_probability(
        self,
        prob: np.ndarray,
        meta: dict,
        threshold: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        thr = float(np.clip(threshold, 0.0, 1.0))
        binary = (prob >= thr)
        if bool(self.cfg.remove_border_connected_air):
            binary = self._remove_border_touching_components(binary)

        ref_image = meta["sitk_image"]

        # Side split by sagittal midline (x-direction).
        x_dim = int(binary.shape[2])
        x_mid = (x_dim - 1) * 0.5
        x_idx = np.arange(x_dim, dtype=np.float32).reshape(1, 1, x_dim)
        left_half = x_idx < x_mid
        right_half = ~left_half

        # Optional neck-cut around midline to disconnect nasal cavity bridges.
        midline_cut = int(max(0, self.cfg.pre_component_midline_cut_vox))
        if midline_cut > 0:
            c = int(round(x_mid))
            x0 = max(0, c - midline_cut)
            x1 = min(x_dim, c + midline_cut + 1)
            binary = binary.copy()
            binary[:, :, x0:x1] = False

        # Optional pre-CC opening to break thin connections before candidate ranking.
        pre_open_mm = float(max(self.cfg.pre_component_opening_mm, 0.0))
        if pre_open_mm > 0.0:
            pre_open_radius = self._radius_from_mm_allow_zero(ref_image, pre_open_mm)
            if max(pre_open_radius) > 0:
                pre_img = sitk.GetImageFromArray(binary.astype(np.uint8))
                pre_img.CopyInformation(ref_image)
                pre_img = sitk.BinaryMorphologicalOpening(pre_img, pre_open_radius)
                binary = sitk.GetArrayFromImage(pre_img).astype(bool, copy=False)

        left_candidates_all: List[Tuple[float, int, int]] = []   # (score, voxels, label_id)
        right_candidates_all: List[Tuple[float, int, int]] = []  # (score, voxels, label_id)
        left_candidates_loc: List[Tuple[float, int, int]] = []
        right_candidates_loc: List[Tuple[float, int, int]] = []

        min_cc_vox = int(max(1, self.cfg.min_component_voxels))
        sx_sigma = max(float(self.cfg.sigma_x_rel), 1e-6)
        sy_sigma = max(float(self.cfg.sigma_y_rel), 1e-6)
        sz_sigma = max(float(self.cfg.sigma_z_rel), 1e-6)
        left_x_target = float(np.clip(self.cfg.target_x_rel_left, 0.0, 1.0))
        right_x_target = float(np.clip(self.cfg.target_x_rel_right, 0.0, 1.0))
        left_x_max = float(np.clip(self.cfg.candidate_left_x_max_rel, 0.0, 1.0))
        right_x_min = float(np.clip(self.cfg.candidate_right_x_min_rel, 0.0, 1.0))
        size_x, size_y, size_z = ref_image.GetSize()

        def _append_candidate(
            vox_total: int,
            ix: int,
            iy: int,
            iz: int,
            label_global: int,
            forced_side: Optional[str] = None,
        ) -> None:
            side = forced_side if forced_side in ("left", "right") else ("left" if ix < x_mid else "right")

            x_rel = float(ix / max(size_x - 1, 1))
            y_rel = float(iy / max(size_y - 1, 1))
            z_rel = float(iz / max(size_z - 1, 1))

            sy = np.exp(-0.5 * ((y_rel - float(self.cfg.target_y_rel)) / sy_sigma) ** 2)
            sz = np.exp(-0.5 * ((z_rel - float(self.cfg.target_z_rel)) / sz_sigma) ** 2)
            if side == "left":
                sx = np.exp(-0.5 * ((x_rel - left_x_target) / sx_sigma) ** 2)
                x_ok = x_rel <= left_x_max
            else:
                sx = np.exp(-0.5 * ((x_rel - right_x_target) / sx_sigma) ** 2)
                x_ok = x_rel >= right_x_min
            loc_score = float(sx * sy * sz)

            w = float(np.clip(self.cfg.component_loc_weight, 0.0, 1.0))
            score = float(vox_total) * ((1.0 - w) + w * loc_score)
            item = (score, int(vox_total), int(label_global))
            in_loc = (
                (y_rel >= float(self.cfg.candidate_y_min_rel))
                and (y_rel <= float(self.cfg.candidate_y_max_rel))
                and (z_rel >= float(self.cfg.candidate_z_min_rel))
                and (z_rel <= float(self.cfg.candidate_z_max_rel))
                and bool(x_ok)
            )

            if side == "left":
                left_candidates_all.append(item)
                if in_loc:
                    left_candidates_loc.append(item)
            else:
                right_candidates_all.append(item)
                if in_loc:
                    right_candidates_loc.append(item)

        if bool(self.cfg.split_components_by_side):
            left_bin = np.logical_and(binary, left_half)
            right_bin = np.logical_and(binary, right_half)

            left_img = sitk.GetImageFromArray(left_bin.astype(np.uint8))
            left_img.CopyInformation(ref_image)
            right_img = sitk.GetImageFromArray(right_bin.astype(np.uint8))
            right_img.CopyInformation(ref_image)

            left_cc = sitk.ConnectedComponent(left_img)
            right_cc = sitk.ConnectedComponent(right_img)
            left_cc_arr = sitk.GetArrayFromImage(left_cc).astype(np.int32, copy=False)
            right_cc_arr = sitk.GetArrayFromImage(right_cc).astype(np.int32, copy=False)

            right_offset = int(left_cc_arr.max())
            cc_arr = left_cc_arr.astype(np.int32, copy=True)
            right_nz = right_cc_arr > 0
            if np.any(right_nz):
                cc_arr[right_nz] = right_cc_arr[right_nz] + int(right_offset)

            left_stat = sitk.LabelShapeStatisticsImageFilter()
            left_stat.Execute(left_cc)
            right_stat = sitk.LabelShapeStatisticsImageFilter()
            right_stat.Execute(right_cc)

            for label_id in left_stat.GetLabels():
                vox_total = int(left_stat.GetNumberOfPixels(label_id))
                if vox_total < min_cc_vox:
                    continue
                cx, cy, cz = left_stat.GetCentroid(int(label_id))
                ix, iy, iz = ref_image.TransformPhysicalPointToIndex((cx, cy, cz))
                _append_candidate(vox_total, int(ix), int(iy), int(iz), int(label_id), forced_side="left")

            for label_id in right_stat.GetLabels():
                vox_total = int(right_stat.GetNumberOfPixels(label_id))
                if vox_total < min_cc_vox:
                    continue
                cx, cy, cz = right_stat.GetCentroid(int(label_id))
                ix, iy, iz = ref_image.TransformPhysicalPointToIndex((cx, cy, cz))
                _append_candidate(
                    vox_total,
                    int(ix),
                    int(iy),
                    int(iz),
                    int(label_id) + int(right_offset),
                    forced_side="right",
                )
        else:
            mask_img = sitk.GetImageFromArray(binary.astype(np.uint8))
            mask_img.CopyInformation(ref_image)
            cc = sitk.ConnectedComponent(mask_img)
            cc_arr = sitk.GetArrayFromImage(cc)
            stat = sitk.LabelShapeStatisticsImageFilter()
            stat.Execute(cc)

            for label_id in stat.GetLabels():
                vox_total = int(stat.GetNumberOfPixels(label_id))
                if vox_total < min_cc_vox:
                    continue

                cx, cy, cz = stat.GetCentroid(int(label_id))
                ix, iy, iz = ref_image.TransformPhysicalPointToIndex((cx, cy, cz))
                _append_candidate(vox_total, int(ix), int(iy), int(iz), int(label_id), forced_side=None)

        if bool(self.cfg.enable_candidate_location_filter):
            left_candidates = left_candidates_loc if len(left_candidates_loc) > 0 else left_candidates_all
            right_candidates = right_candidates_loc if len(right_candidates_loc) > 0 else right_candidates_all
        else:
            left_candidates = left_candidates_all
            right_candidates = right_candidates_all

        left_candidates.sort(key=lambda x: x[0], reverse=True)
        right_candidates.sort(key=lambda x: x[0], reverse=True)

        side_min = int(max(1, self.cfg.side_min_voxels))
        left_pool = [c for c in left_candidates if int(c[1]) >= side_min]
        right_pool = [c for c in right_candidates if int(c[1]) >= side_min]
        if not bool(self.cfg.enforce_bilateral):
            left_pool = left_pool or left_candidates
            right_pool = right_pool or right_candidates
        else:
            # Controlled relaxation: keep bilateral constraint, but avoid empty output
            # when one side falls below side_min after morphology.
            if len(left_pool) == 0 and len(left_candidates) > 0:
                left_pool = left_candidates[:]
            if len(right_pool) == 0 and len(right_candidates) > 0:
                right_pool = right_candidates[:]

        max_pair = int(max(1, self.cfg.max_candidates_per_side))
        left_ranked = left_pool[:max_pair]
        right_ranked = right_pool[:max_pair]

        selected_ids: List[int] = []
        left_selected_ids: set[int] = set()
        right_selected_ids: set[int] = set()

        if len(left_ranked) > 0 and len(right_ranked) > 0:
            max_k = int(max(1, self.cfg.max_components_per_side))

            if max_k == 1:
                ratio_min = float(np.clip(self.cfg.side_balance_min_ratio, 0.0, 1.0))
                ratio_target = float(np.clip(self.cfg.side_balance_target_ratio, 1e-6, 1.0))
                balance_weight = float(max(self.cfg.side_balance_weight, 0.0))

                best_pair: Optional[Tuple[Tuple[float, int, int], Tuple[float, int, int]]] = None
                best_score = float("-inf")

                for left_item in left_ranked:
                    for right_item in right_ranked:
                        left_size = float(max(int(left_item[1]), 1))
                        right_size = float(max(int(right_item[1]), 1))
                        ratio = float(min(left_size, right_size) / max(left_size, right_size))

                        if bool(self.cfg.enforce_bilateral) and ratio < ratio_min:
                            continue

                        pair_score = float(left_item[0] + right_item[0])
                        if ratio < ratio_target:
                            deficit = float((ratio_target - ratio) / max(ratio_target, 1e-6))
                            pair_score *= float(max(0.0, 1.0 - balance_weight * deficit))

                        if pair_score > best_score:
                            best_score = pair_score
                            best_pair = (left_item, right_item)

                if best_pair is None:
                    # Fallback keeps deterministic behavior even if every pair failed hard ratio.
                    best_pair = (left_ranked[0], right_ranked[0])

                left_selected_ids = {int(best_pair[0][2])}
                right_selected_ids = {int(best_pair[1][2])}
            else:
                left_selected_ids = {int(x[2]) for x in left_ranked[:max_k]}
                right_selected_ids = {int(x[2]) for x in right_ranked[:max_k]}

            selected_ids = sorted(left_selected_ids.union(right_selected_ids))
        elif not bool(self.cfg.enforce_bilateral):
            max_k = int(max(1, self.cfg.max_components_per_side))
            if len(left_ranked) > 0:
                left_selected_ids = {int(x[2]) for x in left_ranked[:max_k]}
            if len(right_ranked) > 0:
                right_selected_ids = {int(x[2]) for x in right_ranked[:max_k]}
            selected_ids = sorted(left_selected_ids.union(right_selected_ids))
            if len(selected_ids) == 0:
                merged = (left_candidates + right_candidates)
                if len(merged) > 0:
                    merged.sort(key=lambda x: x[0], reverse=True)
                    selected_ids = [int(merged[0][2])]

        pred_bin = np.isin(cc_arr, selected_ids).astype(np.uint8)

        # Assign side labels: 1 (left), 2 (right) from centroid-side candidates.
        pred_mc = np.zeros(binary.shape, dtype=np.int16)

        left_selected_mask = np.isin(cc_arr, list(left_selected_ids))
        right_selected_mask = np.isin(cc_arr, list(right_selected_ids))
        pred_mc[left_selected_mask] = 1
        pred_mc[right_selected_mask] = 2

        # Morphological clean-up on final candidates (open -> close -> fill holes).
        opening_radius = self._radius_from_mm(ref_image, float(self.cfg.opening_mm))
        closing_radius = self._radius_from_mm(ref_image, float(self.cfg.closing_mm))

        def _morph_clean(mask_arr: np.ndarray) -> np.ndarray:
            img = sitk.GetImageFromArray(mask_arr.astype(np.uint8))
            img.CopyInformation(ref_image)
            if max(opening_radius) > 0:
                img = sitk.BinaryMorphologicalOpening(img, opening_radius)
            if max(closing_radius) > 0:
                img = sitk.BinaryMorphologicalClosing(img, closing_radius)
            if bool(self.cfg.fill_holes_after_morph):
                img = sitk.BinaryFillhole(img)
            return sitk.GetArrayFromImage(img).astype(bool, copy=False)

        left_mask = _morph_clean(pred_mc == 1)
        right_mask = _morph_clean(pred_mc == 2)

        # Enforce left/right assignment with x-midline after morphology.
        left_mask = np.logical_and(left_mask, left_half)
        right_mask = np.logical_and(right_mask, right_half)

        pred_mc = np.zeros(binary.shape, dtype=np.int16)
        pred_mc[left_mask] = 1
        pred_mc[right_mask] = 2
        pred_bin = np.logical_or(left_mask, right_mask).astype(np.uint8)
        return pred_bin, pred_mc

    def evaluate_probability_map(
        self,
        prob: np.ndarray,
        gt_int: np.ndarray,
        meta: dict,
        prob_thr: float,
        sweep_threshold: bool,
        sweep_start: float,
        sweep_end: float,
        sweep_step: float,
        optimize_metric: str,
    ) -> Tuple[dict, List[dict]]:
        gt_int = np.rint(gt_int).astype(np.int16, copy=False)
        gt_bin = gt_int > 0

        if sweep_threshold:
            thresholds = build_thresholds(sweep_start, sweep_end, sweep_step)
        else:
            thresholds = np.asarray([float(prob_thr)], dtype=np.float32)

        rows: List[dict] = []
        best_row: Optional[dict] = None

        for thr in thresholds:
            pred_bin, pred_mc = self.segment_from_probability(prob, meta, float(thr))
            pred_b = pred_bin > 0

            row = {
                "threshold": float(thr),
                "dice": float(dice_coefficient(pred_b, gt_bin)),
                "iou": float(iou_coefficient(pred_b, gt_bin)),
                "pred_voxels": int(pred_b.sum()),
                "target_voxels": int(gt_bin.sum()),
                "intersection_voxels": int(np.logical_and(pred_b, gt_bin).sum()),
                "union_voxels": int(np.logical_or(pred_b, gt_bin).sum()),
            }

            # Label IDs in ground truth may not always match physical left/right ordering.
            # Evaluate both direct and swapped mappings and keep the better balanced score.
            direct = self._compute_label_metrics(pred_mc, gt_int, mapping={1: 1, 2: 2})
            swapped = self._compute_label_metrics(pred_mc, gt_int, mapping={1: 2, 2: 1})
            label_metrics = direct if float(direct.get("dice_balanced", 0.0)) >= float(swapped.get("dice_balanced", 0.0)) else swapped
            row.update(label_metrics)
            if "dice_balanced" not in row:
                row["dice_balanced"] = float(row["dice"])

            rows.append(row)
            if best_row is None or metrics_sort_key(row, optimize_metric) > metrics_sort_key(best_row, optimize_metric):
                best_row = row

        if best_row is None:
            raise RuntimeError("No threshold evaluation row generated.")
        return best_row, rows

    @staticmethod
    def _compute_label_metrics(pred_mc: np.ndarray, gt_int: np.ndarray, mapping: Dict[int, int]) -> dict:
        metrics: Dict[str, float | int] = {}
        dice_values: List[float] = []

        for target_label in (1, 2):
            pred_label = mapping[target_label]
            pred_k = pred_mc == pred_label
            gt_k = gt_int == target_label

            metrics[f"pred_voxels_label{target_label}"] = int(pred_k.sum())
            metrics[f"target_voxels_label{target_label}"] = int(gt_k.sum())
            metrics[f"intersection_voxels_label{target_label}"] = int(np.logical_and(pred_k, gt_k).sum())

            if np.any(gt_k):
                dk = float(dice_coefficient(pred_k, gt_k))
                metrics[f"dice_label{target_label}"] = dk
                dice_values.append(dk)

        metrics["dice_balanced"] = float(np.mean(dice_values)) if len(dice_values) > 0 else 0.0
        return metrics


# -----------------------------
# Backward compatibility adapter
# -----------------------------


class _PriorNetCompat:
    """Compatibility layer for legacy callers expecting `priornet` module API."""

    def __init__(self):
        self._base_cfg = PriorNetConfig()

    @staticmethod
    def _infer_type(path: str) -> str:
        p = str(path).lower()
        if p.endswith(".nrrd") or p.endswith(".nhdr"):
            return "nrrd"
        if p.endswith(".nii") or p.endswith(".nii.gz"):
            return "nifti"
        return "sitk"

    def load_volume(self, path: str) -> Tuple[np.ndarray, dict]:
        model = PriorNet(self._base_cfg)
        vol, meta = model.load_volume(path)
        meta = dict(meta)
        meta["type"] = self._infer_type(path)
        meta.setdefault("affine", None)
        return vol, meta

    def save_volume(self, path: str, vol: np.ndarray, meta: dict) -> None:
        model = PriorNet(self._base_cfg)
        model.save_volume(path, np.asarray(vol, dtype=np.float32), meta)

    @staticmethod
    def get_spacing_mm(meta: dict) -> Tuple[float, float, float]:
        # Return spacing in ndarray axis order (z, y, x) for scipy distance map usage.
        img = meta.get("sitk_image", None)
        if img is None:
            return 1.0, 1.0, 1.0
        sx, sy, sz = img.GetSpacing()
        return float(sz), float(sy), float(sx)

    def compute_air_probability(
        self,
        vol: np.ndarray,
        spacing_mm: Optional[Tuple[float, float, float]] = None,
        affine: Optional[np.ndarray] = None,
        exclude_midline: bool = True,
        air_q: Optional[float] = None,
        prob_gamma: Optional[float] = None,
        **kwargs,
    ) -> np.ndarray:
        cfg = replace(self._base_cfg)
        if air_q is not None:
            cfg.air_quantile = float(np.clip(air_q, 0.0, 1.0))

        model = PriorNet(cfg)

        arr = np.asarray(vol, dtype=np.float32)
        img = sitk.GetImageFromArray(arr)
        if spacing_mm is not None and len(spacing_mm) >= 3:
            # input spacing_mm is expected in (z, y, x), convert to sitk (x, y, z)
            img.SetSpacing((float(spacing_mm[2]), float(spacing_mm[1]), float(spacing_mm[0])))

        meta = {"sitk_image": img, "type": "array", "affine": affine}
        prob, _ = model.compute_probability_map(arr, meta, keep_midline=not bool(exclude_midline))
        if prob_gamma is not None:
            gamma = float(max(prob_gamma, 1e-6))
            prob = np.power(np.clip(prob, 0.0, 1.0), gamma, dtype=np.float32)
        return prob.astype(np.float32, copy=False)


# Expose legacy-compatible object for existing training/inference scripts.
priornet = _PriorNetCompat()
