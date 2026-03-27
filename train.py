#同时利用有标注和无标注数据，进行协同训练
import argparse
import csv
import json
import os
import random
import time
import zlib
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from scipy import ndimage as ndi
from torch.optim import AdamW
from tqdm import tqdm

from dataset import (
    DEFAULT_DATASET_ROOT,
    CaseItem,
    build_array_train_transform,
    build_array_guided_crop_transform,
    build_loader,
    build_train_transform,
    ensure_exists,
    load_split_cases,
    load_validation_cases,
    normalize_intensity_np,
    patch_monai_max_seed,
    summarize_orientation_consistency,
)
from loss import SideAwareDiceCELoss
from PriorNet import PriorNet, PriorNetConfig, priornet
from VISTA3D import build_aorta_vista3d_model

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax
except ImportError:
    dcrf = None
    create_pairwise_bilateral = None
    create_pairwise_gaussian = None
    unary_from_softmax = None

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
DEFAULT_OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "outputs")
HAS_TORCH_AMP_GRADSCALER = hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler")
DETERMINISTIC_LOSS_FALLBACK_WARNED = False


def bytes_to_mib(num_bytes: int) -> float:
    return float(num_bytes) / (1024.0 * 1024.0)


def make_cuda_grad_scaler(enabled: bool):
    if HAS_TORCH_AMP_GRADSCALER:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def cuda_autocast(enabled: bool):
    if HAS_TORCH_AMP_GRADSCALER:
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def is_cuda_oom_error(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return "cuda" in msg and "out of memory" in msg


class PseudoVRAMKeeper:
    def __init__(self, device: torch.device, enabled: bool, chunk_mb: int = 256) -> None:
        self.device = device
        self.enabled = bool(enabled and device.type == "cuda")
        self.chunk_bytes = max(1, int(chunk_mb)) * 1024 * 1024
        self.observed_reserved_bytes = 0
        self.target_reserved_bytes = 0
        self._buffers: List[torch.Tensor] = []

    @property
    def held_bytes(self) -> int:
        return int(sum(int(buf.numel()) * int(buf.element_size()) for buf in self._buffers))

    def observe(self) -> int:
        if not self.enabled:
            return 0
        current = int(torch.cuda.memory_reserved(self.device))
        if current > self.observed_reserved_bytes:
            self.observed_reserved_bytes = current
        return current

    def set_target_from_observed(self, headroom_mb: float) -> int:
        if not self.enabled:
            return 0
        headroom_bytes = max(0, int(float(headroom_mb) * 1024.0 * 1024.0))
        self.target_reserved_bytes = max(0, self.observed_reserved_bytes - headroom_bytes)
        return self.target_reserved_bytes

    def hold(self) -> int:
        if not self.enabled or self.target_reserved_bytes <= 0:
            return 0

        while int(torch.cuda.memory_reserved(self.device)) < self.target_reserved_bytes:
            gap = self.target_reserved_bytes - int(torch.cuda.memory_reserved(self.device))
            chunk_bytes = min(self.chunk_bytes, gap)
            try:
                self._buffers.append(torch.empty((chunk_bytes,), dtype=torch.uint8, device=self.device))
            except RuntimeError as exc:
                if is_cuda_oom_error(exc):
                    break
                raise
        return self.held_bytes

    def release(self) -> int:
        released = self.held_bytes
        self._buffers.clear()
        return released


def run_with_pseudo_vram_retry(fn: Callable[[], object], keeper: PseudoVRAMKeeper, desc: str):
    try:
        return fn()
    except RuntimeError as exc:
        if keeper.enabled and keeper.held_bytes > 0 and is_cuda_oom_error(exc):
            released = keeper.release()
            print(
                f"[VRAM] CUDA OOM during {desc}; released "
                f"{bytes_to_mib(released):.1f} MiB reservation and retry once."
            )
            return fn()
        raise


def set_seed(seed: int) -> None:
    patch_monai_max_seed()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_torch_deterministic_algorithms(enabled: bool, warn_only: Optional[bool] = None) -> None:
    try:
        if warn_only is None:
            torch.use_deterministic_algorithms(enabled)
        else:
            torch.use_deterministic_algorithms(enabled, warn_only=bool(warn_only))
    except TypeError:
        torch.use_deterministic_algorithms(enabled)


def are_torch_deterministic_algorithms_enabled() -> bool:
    fn = getattr(torch, "are_deterministic_algorithms_enabled", None)
    return bool(fn()) if callable(fn) else False


def is_torch_deterministic_warn_only_enabled() -> bool:
    fn = getattr(torch, "is_deterministic_algorithms_warn_only_enabled", None)
    return bool(fn()) if callable(fn) else False


@contextmanager
def temporary_torch_deterministic_algorithms(enabled: bool, warn_only: Optional[bool] = None):
    prev_enabled = are_torch_deterministic_algorithms_enabled()
    prev_warn_only = is_torch_deterministic_warn_only_enabled()
    target_warn_only = prev_warn_only if warn_only is None else bool(warn_only)
    set_torch_deterministic_algorithms(enabled, warn_only=target_warn_only)
    try:
        yield
    finally:
        set_torch_deterministic_algorithms(prev_enabled, warn_only=prev_warn_only)


def configure_torch_runtime(deterministic: bool) -> None:
    deterministic = bool(deterministic)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    set_torch_deterministic_algorithms(deterministic, warn_only=False)


def compute_corefined_loss(
    criterion: DiceCELoss,
    multiclass_logits: torch.Tensor,
    target: torch.Tensor,
    context: str,
) -> torch.Tensor:
    global DETERMINISTIC_LOSS_FALLBACK_WARNED
    if multiclass_logits.is_cuda and are_torch_deterministic_algorithms_enabled():
        if not DETERMINISTIC_LOSS_FALLBACK_WARNED:
            print(
                "Deterministic mode note: temporarily disabling torch deterministic algorithms "
                f"for DiceCELoss CE/NLL on CUDA during {context}."
            )
            DETERMINISTIC_LOSS_FALLBACK_WARNED = True
        with temporary_torch_deterministic_algorithms(enabled=False, warn_only=False):
            return criterion(multiclass_logits, target)
    return criterion(multiclass_logits, target)


def parse_gpu_index(gpu_arg) -> int:
    s = str(gpu_arg).strip()
    if s.lower().startswith("cuda:"):
        s = s.split(":", 1)[1].strip()
    if s == "":
        raise ValueError("Empty --gpu value")
    try:
        idx = int(s)
    except ValueError as exc:
        raise ValueError(f"Invalid --gpu value {gpu_arg!r}. Use integer index like 0/1 or cuda:0/cuda:1.") from exc
    if idx < 0:
        raise ValueError(f"Invalid --gpu value {gpu_arg!r}. GPU index must be >= 0.")
    return idx


def split_ratio_to_tag(split_ratio: str) -> str:
    return str(split_ratio).strip().replace("%", "pct").replace(os.sep, "_").replace("/", "_")


def clamp_prob_np(prob: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(prob, dtype=np.float32), eps, 1.0 - eps)


def logit_np(prob: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = clamp_prob_np(prob, eps=eps)
    return np.log(p) - np.log1p(-p)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def binary_dice(a: np.ndarray, b: np.ndarray) -> float:
    a_bool = np.asarray(a, dtype=bool)
    b_bool = np.asarray(b, dtype=bool)
    inter = int(np.logical_and(a_bool, b_bool).sum())
    den = int(a_bool.sum() + b_bool.sum())
    if den == 0:
        return 1.0
    return float((2.0 * inter) / (den + 1e-8))


def clone_label_tensor(x: object) -> torch.Tensor:
    return torch.as_tensor(x).clone().long()


def ensure_sample_list(samples: object) -> List[Dict[str, object]]:
    if isinstance(samples, list):
        return list(samples)
    return [samples]  # type: ignore[list-item]


def squeeze_single_channel_array(x: object, dtype: Optional[np.dtype] = None) -> np.ndarray:
    arr = np.asarray(x, dtype=dtype)
    if arr.ndim >= 1 and int(arr.shape[0]) == 1:
        return np.asarray(arr[0], dtype=dtype)
    return np.asarray(arr, dtype=dtype)


def build_global_x_coord_map(shape: Sequence[int]) -> np.ndarray:
    shape = tuple(int(v) for v in shape)
    x_dim = int(shape[-1])
    x_rel = np.arange(x_dim, dtype=np.float32).reshape((1,) * (len(shape) - 1) + (x_dim,))
    x_rel = x_rel / max(x_dim - 1, 1)
    return np.broadcast_to(x_rel, shape).astype(np.float32, copy=False)


def infer_vista_patch_probs(
    model: torch.nn.Module,
    image_patch: object,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    image_t = torch.as_tensor(image_patch).unsqueeze(0).to(device).float()
    amp_ctx = cuda_autocast(enabled=True) if use_amp else nullcontext()
    with torch.no_grad():
        with amp_ctx:
            logits = model(image_t)
    vista_multiclass_probs_t, vista_fg_prob_t = vista_output_probs_from_logits(logits)
    return torch.cat((vista_multiclass_probs_t, vista_fg_prob_t), dim=1)[0].detach().cpu().numpy().astype(np.float32)


def get_stable_case_seed(case_id: str) -> int:
    return int(zlib.crc32(str(case_id).encode("utf-8")) & 0xFFFFFFFF)


def reseed_randomizable_transform(transform: object, seed: int) -> None:
    if hasattr(transform, "set_random_state"):
        transform.set_random_state(seed=int(seed))


def generate_patch_pseudo_label(
    model: torch.nn.Module,
    image_patch: object,
    prior_prob_patch: np.ndarray,
    global_x_patch: Optional[np.ndarray],
    args,
    device: torch.device,
    epoch_idx: int,
    use_amp: bool,
    use_prior_segment_for_p: bool,
    use_vista_guidance: bool,
) -> np.ndarray:
    prior_prob_patch = np.asarray(prior_prob_patch, dtype=np.float32)
    prior_fg_thr = float(args.prior_p_thr if bool(use_prior_segment_for_p) else args.default_prob_thr)
    prior_fg_patch = np.asarray(prior_prob_patch >= prior_fg_thr, dtype=bool)

    if not bool(use_vista_guidance):
        return assign_pseudo_classes_from_prior(
            refined_fg=prior_fg_patch,
            args=args,
            global_x_map=global_x_patch,
            preserve_fg_boundary=False,
        ).astype(np.int16, copy=False)

    vista_v_probs = infer_vista_patch_probs(
        model=model,
        image_patch=image_patch,
        device=device,
        use_amp=use_amp,
    )
    guided_fg_prob, vista_fg_prob, fg_seed, bg_seed, guidance_fg = compute_guided_prior_prob(
        prior_prob=prior_prob_patch,
        vista_probs=vista_v_probs,
        args=args,
        epoch_idx=epoch_idx,
    )

    coarse_thr, _ = scan_best_threshold(
        prior_prob=guided_fg_prob,
        guidance_fg=guidance_fg,
        sweep_start=float(args.sweep_start),
        sweep_end=float(args.sweep_end),
        sweep_step=float(args.sweep_step),
        fallback_thr=float(args.default_prob_thr),
        guidance_min_vox=int(args.guidance_min_vox),
    )
    coarse_fg = guided_fg_prob >= float(coarse_thr)
    coarse_fg = select_structured_components(
        fg_mask=coarse_fg,
        refined_prob=guided_fg_prob,
        vista_probs=vista_v_probs,
        vista_fg_prob=vista_fg_prob,
        fg_seed=fg_seed,
        bg_seed=bg_seed,
        global_x_map=global_x_patch,
        args=args,
        epoch_idx=epoch_idx,
    )

    roi_mask = np.asarray(
        coarse_fg | fg_seed | (guided_fg_prob >= float(args.crf_roi_prob_thr)),
        dtype=bool,
    )
    if roi_mask.any():
        roi_mask = ndi.binary_dilation(
            roi_mask,
            structure=ndi.generate_binary_structure(roi_mask.ndim, 1),
            iterations=1,
        )

    refined_prob = run_densecrf_on_roi(
        guided_fg_prob=guided_fg_prob,
        vista_probs=vista_v_probs,
        image_norm=squeeze_single_channel_array(image_patch, dtype=np.float32),
        roi_mask=roi_mask,
        args=args,
    )
    best_thr, _ = scan_best_threshold(
        prior_prob=refined_prob,
        guidance_fg=guidance_fg,
        sweep_start=float(args.sweep_start),
        sweep_end=float(args.sweep_end),
        sweep_step=float(args.sweep_step),
        fallback_thr=float(coarse_thr),
        guidance_min_vox=int(args.guidance_min_vox),
    )

    refined_fg = refined_prob >= float(best_thr)
    refined_fg &= roi_mask
    refined_fg = select_structured_components(
        fg_mask=refined_fg,
        refined_prob=refined_prob,
        vista_probs=vista_v_probs,
        vista_fg_prob=vista_fg_prob,
        fg_seed=fg_seed,
        bg_seed=bg_seed,
        global_x_map=global_x_patch,
        args=args,
        epoch_idx=epoch_idx,
    )
    refined_fg, fg_source = select_pseudo_foreground_mask(
        refined_fg=refined_fg,
        coarse_fg=coarse_fg,
        fg_seed=fg_seed,
        args=args,
    )
    return assign_pseudo_classes_from_vista(
        refined_fg=refined_fg,
        vista_probs=vista_v_probs,
        global_x_map=global_x_patch,
        args=args,
        epoch_idx=epoch_idx,
        preserve_fg_boundary=False,
    ).astype(np.int16, copy=False)


def summarize_validation_patch_pseudo_case(
    model: torch.nn.Module,
    case: CaseItem,
    args,
    device: torch.device,
    epoch_idx: int,
    use_amp: bool,
    prior_eval_model: Optional[PriorNet],
    crop_transform,
) -> Optional[Dict[str, float]]:
    if case.label is None:
        return None

    with torch.no_grad():
        vol, meta = priornet.load_volume(case.image)
        vol = np.asarray(vol, dtype=np.float32)
        prior_p_prob = priornet.compute_air_probability(
            vol,
            spacing_mm=priornet.get_spacing_mm(meta),
            affine=meta.get("affine", None),
            exclude_midline=bool(args.exclude_midline),
            air_ref_method="otsu",
            component_prob_thr=float(args.prior_component_prob_thr),
            prob_gamma=float(args.prior_prob_gamma),
            enable_nonair_branch=True,
            nonair_weight=float(args.nonair_weight),
            nonair_seed_weight=float(args.nonair_seed_weight),
            nonair_center_q=float(args.nonair_center_q),
            nonair_sigma_hu=float(args.nonair_sigma_hu),
            nonair_bone_q=float(args.nonair_bone_q),
            nonair_gate_floor=float(args.nonair_gate_floor),
        ).astype(np.float32)
        image_norm = normalize_intensity_np(vol, args.norm_p_low, args.norm_p_high)
        global_x_map = build_global_x_coord_map(vol.shape)
        gt_label, _ = priornet.load_volume(case.label)
        gt_label = np.asarray(gt_label, dtype=np.int16)
        if gt_label.shape != vol.shape:
            raise ValueError(
                f"Shape mismatch in validation pseudo-patch eval for {case.case_id}: "
                f"image={vol.shape}, gt={gt_label.shape}"
            )

        prior_p_prob_for_pseudo = prior_p_prob
        prior_p_fg_raw = np.asarray(prior_p_prob >= float(args.default_prob_thr), dtype=bool)
        prior_p_fg_strong = prior_p_fg_raw
        if bool(args.use_prior_strong_p):
            seg_model = prior_eval_model if prior_eval_model is not None else PriorNet(PriorNetConfig())
            seg_bin, _ = seg_model.segment_from_probability(
                prob=prior_p_prob,
                meta=meta,
                threshold=float(args.prior_p_thr),
            )
            prior_p_fg_strong = np.asarray(seg_bin > 0, dtype=bool)
            prior_p_prob_for_pseudo = np.where(
                prior_p_fg_strong,
                np.maximum(prior_p_prob, float(args.prior_p_thr)),
                1e-5,
            ).astype(np.float32)

        crop_prior_mask = np.asarray(
            prior_p_fg_strong if bool(args.use_prior_strong_p) else prior_p_fg_raw,
            dtype=np.float32,
        )
        reseed_randomizable_transform(crop_transform, get_stable_case_seed(case.case_id))
        crop_samples = ensure_sample_list(
            crop_transform(
                {
                    "image": image_norm.astype(np.float32),
                    "prior_mask": crop_prior_mask,
                    "prior_prob": prior_p_prob_for_pseudo.astype(np.float32),
                    "global_x": global_x_map.astype(np.float32),
                    "gt_label": gt_label.astype(np.float32),
                }
            )
        )

        patch_count = 0
        sum_dice_cls1 = 0.0
        sum_dice_cls2 = 0.0
        sum_dice_fg = 0.0
        for sample in crop_samples:
            image_patch = torch.as_tensor(sample["image"]).clone().float()
            prior_prob_patch = squeeze_single_channel_array(sample["prior_prob"], dtype=np.float32)
            global_x_patch = squeeze_single_channel_array(sample["global_x"], dtype=np.float32)
            gt_patch = squeeze_single_channel_array(sample["gt_label"], dtype=np.int16)
            pseudo_p_label = generate_patch_pseudo_label(
                model=model,
                image_patch=image_patch,
                prior_prob_patch=prior_prob_patch,
                global_x_patch=global_x_patch,
                args=args,
                device=device,
                epoch_idx=epoch_idx,
                use_amp=use_amp,
                use_prior_segment_for_p=bool(args.use_prior_strong_p),
                use_vista_guidance=True,
            )
            if int((pseudo_p_label > 0).sum()) < int(args.pseudo_train_min_vox):
                continue

            sum_dice_cls1 += binary_dice(pseudo_p_label == 1, gt_patch == 1)
            sum_dice_cls2 += binary_dice(pseudo_p_label == 2, gt_patch == 2)
            sum_dice_fg += binary_dice(pseudo_p_label > 0, gt_patch > 0)
            patch_count += 1

    if patch_count <= 0:
        return None
    return {
        "dice_cls1": float(sum_dice_cls1 / float(patch_count)),
        "dice_cls2": float(sum_dice_cls2 / float(patch_count)),
        "dice_fg": float(sum_dice_fg / float(patch_count)),
    }


def build_corefined_training_samples(
    model: torch.nn.Module,
    case: CaseItem,
    args,
    device: torch.device,
    epoch_idx: int,
    use_amp: bool,
    prior_eval_model: Optional[PriorNet],
    crop_transform,
    use_prior_segment_for_p: bool,
    use_vista_guidance: bool,
    guide_with_gt: bool,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    source = "labeled" if bool(guide_with_gt) else "unlabeled"

    def init_patch_stats() -> Dict[str, object]:
        return {
            "case_id": str(case.case_id),
            "source": str(source),
            "patch_candidates": 0,
            "patch_accepted": 0,
            "fg_vox_list": [],
            "cls1_vox_list": [],
            "cls2_vox_list": [],
            "empty_count": 0,
            "one_side_only_count": 0,
            "reject_min_vox_count": 0,
            "reject_max_vox_count": 0,
            "reject_min_cls_vox_count": 0,
            "reject_require_both_classes_count": 0,
        }

    def summarize_patch_label(label_int: np.ndarray) -> Dict[str, object]:
        label_int = np.asarray(label_int, dtype=np.int16)
        cls1_vox = int((label_int == 1).sum())
        cls2_vox = int((label_int == 2).sum())
        fg_vox = int(cls1_vox + cls2_vox)
        return {
            "fg_vox": int(fg_vox),
            "cls1_vox": int(cls1_vox),
            "cls2_vox": int(cls2_vox),
            "is_empty": bool(fg_vox <= 0),
            "one_side_only": bool((cls1_vox > 0) != (cls2_vox > 0)),
        }

    def record_patch_stats(stats_record: Dict[str, object], patch_stats: Dict[str, object], accepted: bool, reject_reason: str) -> None:
        stats_record["patch_candidates"] = int(stats_record["patch_candidates"]) + 1
        stats_record["patch_accepted"] = int(stats_record["patch_accepted"]) + int(bool(accepted))
        fg_vox_list = stats_record["fg_vox_list"]
        cls1_vox_list = stats_record["cls1_vox_list"]
        cls2_vox_list = stats_record["cls2_vox_list"]
        if isinstance(fg_vox_list, list):
            fg_vox_list.append(int(patch_stats["fg_vox"]))
        if isinstance(cls1_vox_list, list):
            cls1_vox_list.append(int(patch_stats["cls1_vox"]))
        if isinstance(cls2_vox_list, list):
            cls2_vox_list.append(int(patch_stats["cls2_vox"]))
        stats_record["empty_count"] = int(stats_record["empty_count"]) + int(bool(patch_stats["is_empty"]))
        stats_record["one_side_only_count"] = int(stats_record["one_side_only_count"]) + int(bool(patch_stats["one_side_only"]))
        if reject_reason == "min_vox":
            stats_record["reject_min_vox_count"] = int(stats_record["reject_min_vox_count"]) + 1
        elif reject_reason == "max_vox":
            stats_record["reject_max_vox_count"] = int(stats_record["reject_max_vox_count"]) + 1
        elif reject_reason == "min_cls_vox":
            stats_record["reject_min_cls_vox_count"] = int(stats_record["reject_min_cls_vox_count"]) + 1
        elif reject_reason == "require_both_classes":
            stats_record["reject_require_both_classes_count"] = int(stats_record["reject_require_both_classes_count"]) + 1

    def should_accept_pseudo_patch(label_int: np.ndarray) -> Tuple[bool, Dict[str, object], str]:
        patch_stats = summarize_patch_label(label_int)
        reject_reason = ""
        fg_vox = int(patch_stats["fg_vox"])
        cls1_vox = int(patch_stats["cls1_vox"])
        cls2_vox = int(patch_stats["cls2_vox"])
        if fg_vox < int(args.pseudo_train_min_vox):
            reject_reason = "min_vox"
        elif int(args.pseudo_train_max_vox) > 0 and fg_vox > int(args.pseudo_train_max_vox):
            reject_reason = "max_vox"
        elif bool(args.pseudo_require_both_classes) and (cls1_vox <= 0 or cls2_vox <= 0):
            reject_reason = "require_both_classes"
        elif int(args.pseudo_train_min_cls_vox) > 0:
            min_cls_vox = int(args.pseudo_train_min_cls_vox)
            if (0 < cls1_vox < min_cls_vox) or (0 < cls2_vox < min_cls_vox):
                reject_reason = "min_cls_vox"
        return (reject_reason == ""), patch_stats, reject_reason

    patch_stats = init_patch_stats()
    model.eval()

    with torch.no_grad():
        vol, meta = priornet.load_volume(case.image)
        vol = np.asarray(vol, dtype=np.float32)
        spacing_mm = priornet.get_spacing_mm(meta)
        prior_p_prob = priornet.compute_air_probability(
            vol,
            spacing_mm=spacing_mm,
            affine=meta.get("affine", None),
            exclude_midline=bool(args.exclude_midline),
            air_ref_method="otsu",
            component_prob_thr=float(args.prior_component_prob_thr),
            prob_gamma=float(args.prior_prob_gamma),
            enable_nonair_branch=True,
            nonair_weight=float(args.nonair_weight),
            nonair_seed_weight=float(args.nonair_seed_weight),
            nonair_center_q=float(args.nonair_center_q),
            nonair_sigma_hu=float(args.nonair_sigma_hu),
            nonair_bone_q=float(args.nonair_bone_q),
            nonair_gate_floor=float(args.nonair_gate_floor),
        ).astype(np.float32)
        image_norm = normalize_intensity_np(vol, args.norm_p_low, args.norm_p_high)
        global_x_map = build_global_x_coord_map(vol.shape)

        prior_p_prob_for_pseudo = prior_p_prob
        prior_p_fg_raw = np.asarray(prior_p_prob >= float(args.default_prob_thr), dtype=bool)
        prior_p_fg_strong = prior_p_fg_raw
        if bool(use_prior_segment_for_p):
            seg_model = prior_eval_model if prior_eval_model is not None else PriorNet(PriorNetConfig())
            seg_bin, _ = seg_model.segment_from_probability(
                prob=prior_p_prob,
                meta=meta,
                threshold=float(args.prior_p_thr),
            )
            prior_p_fg_strong = np.asarray(seg_bin > 0, dtype=bool)
            prior_p_prob_for_pseudo = np.where(
                prior_p_fg_strong,
                np.maximum(prior_p_prob, float(args.prior_p_thr)),
                1e-5,
            ).astype(np.float32)

        if bool(guide_with_gt):
            if case.label is None:
                raise ValueError(f"GT-guided co-refinement requires label for case {case.case_id}.")
            gt_label, _ = priornet.load_volume(case.label)
            gt_label = np.asarray(gt_label, dtype=np.int16)
            if gt_label.shape != vol.shape:
                raise ValueError(
                    f"Shape mismatch in GT-guided co-refinement for {case.case_id}: "
                    f"image={vol.shape}, gt={gt_label.shape}"
                )

            crop_samples = ensure_sample_list(
                crop_transform(
                    {
                        "image": image_norm.astype(np.float32),
                        "prior_mask": gt_label.astype(np.float32),
                        "prior_prob": prior_p_prob_for_pseudo.astype(np.float32),
                        "global_x": global_x_map.astype(np.float32),
                        "gt_label": gt_label.astype(np.float32),
                    }
                )
            )

            out: List[Dict[str, object]] = []
            for sample in crop_samples:
                gt_patch = squeeze_single_channel_array(sample["gt_label"], dtype=np.int16)
                gt_patch_stats = summarize_patch_label(gt_patch)
                record_patch_stats(patch_stats, gt_patch_stats, accepted=True, reject_reason="")
                out.append(
                    {
                        "image": torch.as_tensor(sample["image"]).clone().float(),
                        "label": clone_label_tensor(gt_patch),
                        "source": "labeled",
                    }
                )
            return out, patch_stats

        crop_guide = np.asarray(
            prior_p_fg_strong if bool(use_prior_segment_for_p) else prior_p_fg_raw,
            dtype=np.float32,
        )
        crop_samples = ensure_sample_list(
            crop_transform(
                {
                    "image": image_norm.astype(np.float32),
                    "prior_mask": crop_guide,
                    "prior_prob": prior_p_prob_for_pseudo.astype(np.float32),
                    "global_x": global_x_map.astype(np.float32),
                    "gt_label": np.zeros_like(prior_p_prob_for_pseudo, dtype=np.float32),
                }
            )
        )

        out: List[Dict[str, object]] = []
        for sample in crop_samples:
            image_patch = torch.as_tensor(sample["image"]).clone().float()
            prior_prob_patch = squeeze_single_channel_array(sample["prior_prob"], dtype=np.float32)
            global_x_patch = squeeze_single_channel_array(sample["global_x"], dtype=np.float32)
            pseudo_p_label = generate_patch_pseudo_label(
                model=model,
                image_patch=image_patch,
                prior_prob_patch=prior_prob_patch,
                global_x_patch=global_x_patch,
                args=args,
                device=device,
                epoch_idx=epoch_idx,
                use_amp=use_amp,
                use_prior_segment_for_p=bool(use_prior_segment_for_p),
                use_vista_guidance=bool(use_vista_guidance),
            )
            accept_patch, pseudo_patch_stats, reject_reason = should_accept_pseudo_patch(pseudo_p_label)
            record_patch_stats(patch_stats, pseudo_patch_stats, accepted=accept_patch, reject_reason=reject_reason)
            if not accept_patch:
                continue
            out.append(
                {
                    "image": image_patch,
                    "label": clone_label_tensor(pseudo_p_label),
                    "source": "unlabeled",
                }
            )
        return out, patch_stats


def split_vista_output_logits(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 5 or logits.shape[1] < 3:
        raise ValueError(f"Expected VISTA logits with shape [B, >=3, H, W, D], got {tuple(logits.shape)}.")
    multiclass_logits = logits[:, :3]
    if logits.shape[1] >= 4:
        fg_binary_logit = logits[:, 3:4]
    else:
        fg_binary_logit = torch.logsumexp(multiclass_logits[:, 1:3], dim=1, keepdim=True) - multiclass_logits[:, 0:1]
    return multiclass_logits, fg_binary_logit


def vista_output_probs_from_logits(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    multiclass_logits, fg_binary_logit = split_vista_output_logits(logits)
    multiclass_probs = torch.softmax(multiclass_logits, dim=1)
    fg_prob = torch.sigmoid(fg_binary_logit)
    return multiclass_probs, fg_prob


def vista_fg_prob_from_probs(vista_probs: np.ndarray) -> np.ndarray:
    vista_probs = np.asarray(vista_probs, dtype=np.float32)
    if vista_probs.ndim < 1 or vista_probs.shape[0] < 3:
        raise ValueError(f"Expected VISTA probability map with shape [>=3, ...], got {vista_probs.shape}.")
    if vista_probs.shape[0] >= 4:
        return clamp_prob_np(vista_probs[3])
    return clamp_prob_np(vista_probs[1] + vista_probs[2])


def vista_pred_from_probs(vista_probs: np.ndarray) -> np.ndarray:
    vista_probs = np.asarray(vista_probs, dtype=np.float32)
    if vista_probs.ndim < 1 or vista_probs.shape[0] < 3:
        raise ValueError(f"Expected VISTA probability map with shape [>=3, ...], got {vista_probs.shape}.")
    return np.argmax(vista_probs[:3], axis=0).astype(np.int16, copy=False)


def build_thresholds(start: float, end: float, step: float) -> np.ndarray:
    s = float(start)
    e = float(end)
    st = float(step)
    if st <= 0:
        raise ValueError("sweep_step must be > 0")
    if e < s:
        s, e = e, s
    n = int(np.floor((e - s) / st + 1e-8)) + 1
    vals = s + np.arange(n, dtype=np.float32) * st
    vals = vals[(vals >= s - 1e-6) & (vals <= e + 1e-6)]
    if vals.size == 0:
        vals = np.asarray([s], dtype=np.float32)
    return vals.astype(np.float32)


def scan_best_threshold(
    prior_prob: np.ndarray,
    guidance_fg: np.ndarray,
    sweep_start: float,
    sweep_end: float,
    sweep_step: float,
    fallback_thr: float,
    guidance_min_vox: int,
) -> Tuple[float, float]:
    guidance_fg = np.asarray(guidance_fg, dtype=bool)
    if int(guidance_fg.sum()) < int(guidance_min_vox):
        return float(fallback_thr), float("nan")

    thresholds = build_thresholds(sweep_start, sweep_end, sweep_step)
    best_thr = float(fallback_thr)
    best_score = -1.0

    for thr in thresholds:
        pred_fg = np.asarray(prior_prob >= float(thr), dtype=bool)
        score = binary_dice(pred_fg, guidance_fg)
        if (score > best_score + 1e-8) or (abs(score - best_score) <= 1e-8 and thr < best_thr):
            best_score = float(score)
            best_thr = float(thr)

    return float(best_thr), float(best_score)


def remove_small_components_per_class(label_int: np.ndarray, min_size: int) -> np.ndarray:
    if int(min_size) <= 0:
        return label_int

    out = label_int.astype(np.int16, copy=True)
    for cls in (1, 2):
        mask = out == cls
        if int(mask.sum()) == 0:
            continue
        lab, n_cc = ndi.label(mask)
        if n_cc <= 0:
            continue
        cnt = np.bincount(lab.ravel())
        cnt[0] = 0
        keep_ids = np.where(cnt >= int(min_size))[0]
        if keep_ids.size == 0:
            out[mask] = 0
            continue
        drop = (lab > 0) & (~np.isin(lab, keep_ids))
        out[drop] = 0
    return out


def ensure_densecrf_available() -> None:
    if dcrf is None or create_pairwise_bilateral is None or create_pairwise_gaussian is None or unary_from_softmax is None:
        raise ImportError(
            "pydensecrf is not available in the current Python environment. "
            "Run train.py inside the `tooth` environment where pydensecrf is installed."
        )


def get_pseudo_fg_reliance(epoch: int, args) -> float:
    start_epoch = max(1, int(args.pseudo_start_epoch))
    anneal_epochs = max(1, int(args.pseudo_fg_anneal_epochs))
    if epoch <= start_epoch:
        progress = 0.0
    else:
        progress = min(1.0, float(epoch - start_epoch) / float(anneal_epochs))
    alpha = float(args.pseudo_fg_vista_reliance_start) + progress * (
        float(args.pseudo_fg_vista_reliance_end) - float(args.pseudo_fg_vista_reliance_start)
    )
    return float(np.clip(alpha, 0.0, 1.0))


def compute_guided_prior_prob(
    prior_prob: np.ndarray,
    vista_probs: np.ndarray,
    args,
    epoch_idx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prior_prob = clamp_prob_np(prior_prob)
    vista_fg_prob = vista_fg_prob_from_probs(vista_probs)
    vista_bg_prob = clamp_prob_np(vista_probs[0])
    fg_reliance = get_pseudo_fg_reliance(epoch_idx, args)

    fg_seed_thr = min(0.995, float(args.vista_fg_seed_thr) + (1.0 - fg_reliance) * float(args.pseudo_fg_seed_thr_bonus))
    bg_seed_thr = min(0.999, float(args.vista_bg_seed_thr) + (1.0 - fg_reliance) * float(args.pseudo_bg_seed_thr_bonus))
    fg_seed = vista_fg_prob >= fg_seed_thr
    bg_seed = vista_bg_prob >= bg_seed_thr
    prior_guidance = prior_prob >= float(args.default_prob_thr)
    guidance_fg = fg_seed
    if int(guidance_fg.sum()) < int(args.guidance_min_vox):
        if fg_reliance < 0.5 and int(prior_guidance.sum()) >= int(args.guidance_min_vox):
            guidance_fg = prior_guidance
        else:
            guidance_fg = vista_pred_from_probs(vista_probs) > 0

    guided_logit = logit_np(prior_prob)
    guided_logit += fg_reliance * float(args.vista_guidance_weight) * logit_np(vista_fg_prob)
    guided_logit[fg_seed] += fg_reliance * float(args.vista_seed_boost)
    guided_logit[bg_seed] -= fg_reliance * float(args.vista_seed_boost)
    guided_logit *= float(args.crf_update_factor)

    guided_fg_prob = clamp_prob_np(sigmoid_np(guided_logit))
    return guided_fg_prob, vista_fg_prob, fg_seed, bg_seed, guidance_fg


def compute_roi_bbox(mask: np.ndarray, margin: int) -> Optional[Tuple[slice, ...]]:
    coords = np.argwhere(np.asarray(mask, dtype=bool))
    if coords.size == 0:
        return None

    bbox: List[slice] = []
    for dim in range(coords.shape[1]):
        lo = max(0, int(coords[:, dim].min()) - int(margin))
        hi = min(mask.shape[dim], int(coords[:, dim].max()) + 1 + int(margin))
        bbox.append(slice(lo, hi))
    return tuple(bbox)


def run_densecrf_on_roi(
    guided_fg_prob: np.ndarray,
    vista_probs: np.ndarray,
    image_norm: np.ndarray,
    roi_mask: np.ndarray,
    args,
) -> np.ndarray:
    ensure_densecrf_available()

    bbox = compute_roi_bbox(roi_mask, margin=int(args.crf_roi_margin))
    if bbox is None:
        return np.zeros_like(guided_fg_prob, dtype=np.float32)

    guided_crop = clamp_prob_np(guided_fg_prob[bbox].astype(np.float32))
    roi_crop = np.asarray(roi_mask[bbox], dtype=bool)
    ref_crop = np.stack(
        [
            np.asarray(image_norm[bbox], dtype=np.float32),
            np.asarray(vista_fg_prob_from_probs(vista_probs)[bbox], dtype=np.float32),
            np.asarray(vista_probs[1][bbox], dtype=np.float32),
            np.asarray(vista_probs[2][bbox], dtype=np.float32),
        ],
        axis=-1,
    ).astype(np.float32)

    guided_crop[~roi_crop] = 1e-5
    unary_prob = np.stack([1.0 - guided_crop, guided_crop], axis=0).astype(np.float32)
    unary = unary_from_softmax(unary_prob)

    crf = dcrf.DenseCRF(int(np.prod(guided_crop.shape)), 2)
    crf.setUnaryEnergy(unary)

    spatial_sigmas = tuple(float(args.crf_gaussian_spatial_sigma) for _ in guided_crop.shape)
    bilateral_spatial_sigmas = tuple(float(args.crf_bilateral_spatial_sigma) for _ in guided_crop.shape)
    bilateral_color_sigmas = tuple(float(args.crf_bilateral_color_sigma) for _ in range(ref_crop.shape[-1]))

    gaussian_feats = create_pairwise_gaussian(sdims=spatial_sigmas, shape=guided_crop.shape)
    bilateral_feats = create_pairwise_bilateral(
        sdims=bilateral_spatial_sigmas,
        schan=bilateral_color_sigmas,
        img=ref_crop,
        chdim=ref_crop.ndim - 1,
    )

    crf.addPairwiseEnergy(gaussian_feats, compat=float(args.crf_gaussian_weight))
    crf.addPairwiseEnergy(bilateral_feats, compat=float(args.crf_bilateral_weight))

    q = np.array(crf.inference(int(args.crf_iters)), dtype=np.float32).reshape((2,) + guided_crop.shape)
    refined_crop = clamp_prob_np(q[1])
    refined_crop[~roi_crop] = 0.0

    refined_full = np.zeros_like(guided_fg_prob, dtype=np.float32)
    refined_full[bbox] = refined_crop
    return refined_full


def build_lateral_masks(
    shape: Sequence[int],
    left_cut: float,
    right_cut: float,
    x_rel_map: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x_rel_map is None:
        x_dim = int(shape[-1])
        x_rel = np.arange(x_dim, dtype=np.float32).reshape((1,) * (len(shape) - 1) + (x_dim,))
        x_rel = x_rel / max(x_dim - 1, 1)
        x_rel_map = np.broadcast_to(x_rel, shape)
    else:
        x_rel_map = np.asarray(x_rel_map, dtype=np.float32)
        if x_rel_map.shape != tuple(int(v) for v in shape):
            raise ValueError(f"x_rel_map shape mismatch: expected {tuple(shape)}, got {x_rel_map.shape}")
    # NasalSeg labels_sinus uses the larger x side as class-1 side.
    left_mask = np.asarray(x_rel_map > float(right_cut), dtype=bool)
    right_mask = np.asarray(x_rel_map < float(left_cut), dtype=bool)
    center_mask = ~(left_mask | right_mask)
    return left_mask, center_mask, right_mask


def get_pseudo_class_reliance(epoch: int, args) -> float:
    start_epoch = max(1, int(args.pseudo_start_epoch))
    anneal_epochs = max(1, int(args.pseudo_class_anneal_epochs))
    if epoch <= start_epoch:
        progress = 0.0
    else:
        progress = min(1.0, float(epoch - start_epoch) / float(anneal_epochs))
    alpha = float(args.pseudo_vista_reliance_start) + progress * (
        float(args.pseudo_vista_reliance_end) - float(args.pseudo_vista_reliance_start)
    )
    return float(np.clip(alpha, 0.0, 1.0))


def select_structured_components(
    fg_mask: np.ndarray,
    refined_prob: np.ndarray,
    vista_probs: np.ndarray,
    vista_fg_prob: np.ndarray,
    fg_seed: np.ndarray,
    bg_seed: np.ndarray,
    global_x_map: Optional[np.ndarray],
    args,
    epoch_idx: int,
) -> np.ndarray:
    fg_mask = np.asarray(fg_mask, dtype=bool)
    if not fg_mask.any():
        return fg_mask

    conn = ndi.generate_binary_structure(fg_mask.ndim, 1)
    fg_mask = ndi.binary_closing(fg_mask, structure=conn, iterations=1)
    fg_mask = ndi.binary_fill_holes(fg_mask)

    comp_arr, n_cc = ndi.label(fg_mask, structure=conn)
    if n_cc <= 0:
        return np.zeros_like(fg_mask, dtype=bool)

    cls_seed_1 = vista_probs[1] >= float(args.vista_cls_seed_thr)
    cls_seed_2 = vista_probs[2] >= float(args.vista_cls_seed_thr)
    left_mask, center_mask, right_mask = build_lateral_masks(
        fg_mask.shape,
        left_cut=float(args.pseudo_midline_low),
        right_cut=float(args.pseudo_midline_high),
        x_rel_map=global_x_map,
    )

    kept_ids: List[int] = []
    best_id = 0
    best_score = float("-inf")
    best_side_ids: Dict[str, int] = {"left": 0, "right": 0}
    best_side_scores: Dict[str, float] = {"left": float("-inf"), "right": float("-inf")}
    component_side: Dict[int, str] = {}
    fg_reliance = get_pseudo_fg_reliance(epoch_idx, args)
    class_reliance = get_pseudo_class_reliance(epoch_idx, args)
    cls_support_weight = 0.15 * class_reliance
    vista_support_weight = 0.25 * fg_reliance
    fg_seed_weight = 0.20 * fg_reliance
    bg_conflict_weight = 0.20 * fg_reliance

    for cc_id in range(1, n_cc + 1):
        comp_mask = comp_arr == cc_id
        comp_size = int(comp_mask.sum())
        if comp_size <= 0:
            continue

        mean_prob = float(np.mean(refined_prob[comp_mask]))
        vista_support = float(np.mean(vista_fg_prob[comp_mask]))
        fg_seed_ratio = float(np.mean(fg_seed[comp_mask]))
        bg_conflict = float(np.mean(bg_seed[comp_mask]))
        cls_support = float(max(np.mean(cls_seed_1[comp_mask]), np.mean(cls_seed_2[comp_mask])))
        left_frac = float(np.mean(left_mask[comp_mask]))
        right_frac = float(np.mean(right_mask[comp_mask]))
        center_frac = float(np.mean(center_mask[comp_mask]))
        dominant_side = "left" if left_frac >= right_frac else "right"
        component_side[int(cc_id)] = dominant_side

        score = (
            0.40 * mean_prob
            + vista_support_weight * vista_support
            + fg_seed_weight * fg_seed_ratio
            + cls_support_weight * cls_support
            - bg_conflict_weight * bg_conflict
            - 0.10 * center_frac
        )

        if score > best_score:
            best_score = score
            best_id = int(cc_id)
        if score > best_side_scores[dominant_side]:
            best_side_scores[dominant_side] = score
            best_side_ids[dominant_side] = int(cc_id)

        keep = False
        if comp_size >= int(args.pseudo_min_component):
            if score >= float(args.pseudo_structure_score_thr):
                keep = True
            if fg_seed_ratio > 0.0 and vista_support >= float(args.vista_fg_seed_thr) * 0.5:
                keep = True
        elif fg_seed_ratio >= 0.25 and comp_size >= max(16, int(args.pseudo_min_component) // 2):
            keep = True

        if keep:
            kept_ids.append(int(cc_id))

    kept_set = set(kept_ids)
    for side in ("left", "right"):
        has_side = any(component_side.get(cc_id) == side for cc_id in kept_set)
        side_best_id = int(best_side_ids[side])
        side_best_score = float(best_side_scores[side])
        if (not has_side) and side_best_id > 0:
            relaxed_thr = min(float(args.pseudo_structure_score_thr), float(args.pseudo_side_relaxed_score_thr))
            if side_best_score >= relaxed_thr:
                kept_set.add(side_best_id)

    kept_ids = sorted(kept_set)
    if len(kept_ids) == 0 and best_id > 0:
        kept_ids = [best_id]

    kept = np.isin(comp_arr, kept_ids)
    kept = ndi.binary_fill_holes(kept)
    return np.asarray(kept, dtype=bool)


def select_pseudo_foreground_mask(
    refined_fg: np.ndarray,
    coarse_fg: np.ndarray,
    fg_seed: np.ndarray,
    args,
) -> Tuple[np.ndarray, str]:
    min_vox = int(args.pseudo_fallback_min_vox)
    conn = ndi.generate_binary_structure(refined_fg.ndim, 1)
    seed_fg = ndi.binary_dilation(np.asarray(fg_seed, dtype=bool), structure=conn, iterations=1)
    seed_fg = ndi.binary_fill_holes(seed_fg)

    candidates = [
        ("refined", np.asarray(refined_fg, dtype=bool)),
        ("coarse", np.asarray(coarse_fg, dtype=bool)),
        ("seed", np.asarray(seed_fg, dtype=bool)),
    ]

    best_name = "refined"
    best_mask = candidates[0][1]
    best_vox = int(best_mask.sum())
    for name, mask in candidates:
        vox = int(mask.sum())
        if vox > best_vox:
            best_name = name
            best_mask = mask
            best_vox = vox
        if vox >= min_vox:
            return mask, name
    return best_mask, best_name


def assign_pseudo_classes_from_vista(
    refined_fg: np.ndarray,
    vista_probs: np.ndarray,
    global_x_map: Optional[np.ndarray],
    args,
    epoch_idx: int,
    preserve_fg_boundary: bool = False,
) -> np.ndarray:
    out = np.zeros(refined_fg.shape, dtype=np.int16)
    if not np.any(refined_fg):
        return out

    comp_arr, n_cc = ndi.label(np.asarray(refined_fg, dtype=bool))
    cls_prob = vista_probs[1:3]
    cls_seed_1 = cls_prob[0] >= float(args.vista_cls_seed_thr)
    cls_seed_2 = cls_prob[1] >= float(args.vista_cls_seed_thr)

    left_mask, _, right_mask = build_lateral_masks(
        refined_fg.shape,
        left_cut=float(args.pseudo_midline_low),
        right_cut=float(args.pseudo_midline_high),
        x_rel_map=global_x_map,
    )
    if global_x_map is None:
        x_dim = refined_fg.shape[-1]
        x_rel = np.arange(x_dim, dtype=np.float32).reshape((1,) * (refined_fg.ndim - 1) + (x_dim,))
        x_rel = x_rel / max(x_dim - 1, 1)
        global_x_map = np.broadcast_to(x_rel, refined_fg.shape)
    else:
        global_x_map = np.asarray(global_x_map, dtype=np.float32)
        if global_x_map.shape != refined_fg.shape:
            raise ValueError(f"global_x_map shape mismatch: expected {refined_fg.shape}, got {global_x_map.shape}")
    left_half = np.asarray(global_x_map < 0.5, dtype=bool)
    right_half = np.asarray(global_x_map >= 0.5, dtype=bool)
    class_reliance = get_pseudo_class_reliance(epoch_idx, args)
    side_force_thr = float(np.clip(float(args.pseudo_side_assign_thr) - 0.10 + 0.20 * class_reliance, 0.35, 0.85))
    vista_margin_thr = float(
        max(float(args.pseudo_component_class_margin), 0.25 - 0.20 * class_reliance)
    )

    for cc_id in range(1, n_cc + 1):
        comp_mask = comp_arr == cc_id
        if not np.any(comp_mask):
            continue

        left_part = comp_mask & left_half
        right_part = comp_mask & right_half
        if int(left_part.sum()) > 0 and int(right_part.sum()) > 0:
            out[left_part] = 1
            out[right_part] = 2
            continue

        left_frac = float(np.mean(left_mask[comp_mask]))
        right_frac = float(np.mean(right_mask[comp_mask]))
        if left_frac >= side_force_thr and left_frac > right_frac:
            out[comp_mask] = 1
            continue
        if right_frac >= side_force_thr and right_frac > left_frac:
            out[comp_mask] = 2
            continue

        score_1 = float(np.mean(cls_prob[0][comp_mask])) + 0.5 * float(np.mean(cls_seed_1[comp_mask]))
        score_2 = float(np.mean(cls_prob[1][comp_mask])) + 0.5 * float(np.mean(cls_seed_2[comp_mask]))
        if abs(score_1 - score_2) >= vista_margin_thr:
            out[comp_mask] = 1 if score_1 >= score_2 else 2
        else:
            fused_1 = (1.0 - class_reliance) * left_frac + class_reliance * score_1
            fused_2 = (1.0 - class_reliance) * right_frac + class_reliance * score_2
            if abs(fused_1 - fused_2) >= float(args.pseudo_component_class_margin):
                out[comp_mask] = 1 if fused_1 >= fused_2 else 2
            elif class_reliance < 0.5:
                out[comp_mask] = 1 if left_frac >= right_frac else 2
            else:
                out[comp_mask] = 1 if score_1 >= score_2 else 2

    if not bool(preserve_fg_boundary):
        out = remove_small_components_per_class(out, min_size=int(args.pseudo_min_component))
    return out


def assign_pseudo_classes_from_prior(
    refined_fg: np.ndarray,
    global_x_map: Optional[np.ndarray],
    args,
    preserve_fg_boundary: bool = False,
) -> np.ndarray:
    out = np.zeros(refined_fg.shape, dtype=np.int16)
    if not np.any(refined_fg):
        return out

    fg_mask = np.asarray(refined_fg, dtype=bool)
    conn = ndi.generate_binary_structure(fg_mask.ndim, 1)
    fg_mask = ndi.binary_closing(fg_mask, structure=conn, iterations=1)
    fg_mask = ndi.binary_fill_holes(fg_mask)

    comp_arr, n_cc = ndi.label(fg_mask, structure=conn)
    left_mask, _, right_mask = build_lateral_masks(
        fg_mask.shape,
        left_cut=float(args.pseudo_midline_low),
        right_cut=float(args.pseudo_midline_high),
        x_rel_map=global_x_map,
    )
    if global_x_map is None:
        x_dim = fg_mask.shape[-1]
        x_rel = np.arange(x_dim, dtype=np.float32).reshape((1,) * (fg_mask.ndim - 1) + (x_dim,))
        x_rel = x_rel / max(x_dim - 1, 1)
        global_x_map = np.broadcast_to(x_rel, fg_mask.shape)
    else:
        global_x_map = np.asarray(global_x_map, dtype=np.float32)
        if global_x_map.shape != fg_mask.shape:
            raise ValueError(f"global_x_map shape mismatch: expected {fg_mask.shape}, got {global_x_map.shape}")
    left_half = np.asarray(global_x_map < 0.5, dtype=bool)
    right_half = np.asarray(global_x_map >= 0.5, dtype=bool)

    for cc_id in range(1, n_cc + 1):
        comp_mask = comp_arr == cc_id
        if not np.any(comp_mask):
            continue

        left_part = comp_mask & left_half
        right_part = comp_mask & right_half
        if int(left_part.sum()) > 0 and int(right_part.sum()) > 0:
            out[left_part] = 1
            out[right_part] = 2
            continue

        left_frac = float(np.mean(left_mask[comp_mask]))
        right_frac = float(np.mean(right_mask[comp_mask]))
        out[comp_mask] = 1 if left_frac >= right_frac else 2

    if not bool(preserve_fg_boundary):
        out = remove_small_components_per_class(out, min_size=int(args.pseudo_min_component))
    return out


def get_current_corefined_weight(epoch: int, args) -> float:
    pseudo_start_epoch = max(1, int(args.pseudo_start_epoch))
    if epoch < pseudo_start_epoch:
        return 0.0
    return float(args.corefined_weight)


def use_unlabeled_corefined(epoch: int, args) -> bool:
    if bool(getattr(args, "gt_only", False)):
        return False
    return int(epoch) >= max(1, int(getattr(args, "unlabeled_start_epoch", 1)))


def get_source_corefined_weight(source: str, args) -> float:
    if str(source) == "labeled":
        return float(getattr(args, "labeled_loss_weight", 1.0))
    if str(source) == "unlabeled":
        return float(getattr(args, "unlabeled_loss_weight", 1.0))
    return 1.0


def get_current_lr(epoch: int, args) -> float:
    base_lr = float(args.lr)
    warmup_epochs = max(0, int(args.lr_warmup_epochs))
    max_epochs = max(1, int(args.max_epochs))

    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return base_lr * (float(epoch) / float(warmup_epochs))

    if max_epochs <= warmup_epochs:
        return base_lr

    cosine_total = max(1, max_epochs - warmup_epochs)
    cosine_progress = min(1.0, max(0.0, float(epoch - warmup_epochs) / float(cosine_total)))
    cosine_scale = 0.5 * (1.0 + float(np.cos(np.pi * cosine_progress)))
    return base_lr * cosine_scale


def get_unsup_accum_steps(num_labeled: int, num_unlabeled: int, args) -> int:
    manual_steps = int(args.unsup_accum_steps)
    if manual_steps > 0:
        return max(1, manual_steps)

    if int(num_labeled) <= 0 or int(num_unlabeled) <= 0:
        return 1

    batchsize = max(1, int(args.batchsize))
    target_ratio = (float(num_unlabeled) * float(batchsize)) / (float(num_labeled) * float(batchsize))
    accum_steps = int(np.ceil(target_ratio))
    return max(1, min(accum_steps, int(args.max_unsup_accum_steps)))


def binary_hd95(a: np.ndarray, b: np.ndarray, spacing_mm: Sequence[float]) -> float:
    a_bool = np.asarray(a, dtype=bool)
    b_bool = np.asarray(b, dtype=bool)

    if not a_bool.any() and not b_bool.any():
        return 0.0
    if not a_bool.any() or not b_bool.any():
        return float("inf")

    ndim = int(a_bool.ndim)
    conn = ndi.generate_binary_structure(ndim, 1)
    a_erode = ndi.binary_erosion(a_bool, structure=conn, border_value=0)
    b_erode = ndi.binary_erosion(b_bool, structure=conn, border_value=0)
    a_surf = np.logical_xor(a_bool, a_erode)
    b_surf = np.logical_xor(b_bool, b_erode)

    if not a_surf.any():
        a_surf = a_bool
    if not b_surf.any():
        b_surf = b_bool

    spacing = tuple(float(spacing_mm[i]) if i < len(spacing_mm) else 1.0 for i in range(ndim))
    dist_to_b = ndi.distance_transform_edt(~b_bool, sampling=spacing)
    dist_to_a = ndi.distance_transform_edt(~a_bool, sampling=spacing)
    d_ab = dist_to_b[a_surf]
    d_ba = dist_to_a[b_surf]

    if d_ab.size == 0 or d_ba.size == 0:
        return float("inf")
    d = np.concatenate([d_ab, d_ba], axis=0)
    return float(np.percentile(d, 95.0))


def relabel_foreground_by_geometry(
    pred: np.ndarray,
    class1_is_high_x: bool,
    split_ratio: float = 0.5,
) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.int16)
    fg_mask = pred > 0
    out = np.zeros_like(pred, dtype=np.int16)
    if not np.any(fg_mask):
        return out

    conn = ndi.generate_binary_structure(fg_mask.ndim, 1)
    comp_arr, n_cc = ndi.label(fg_mask, structure=conn)

    x_dim = int(fg_mask.shape[-1])
    x_rel = np.arange(x_dim, dtype=np.float32).reshape((1,) * (fg_mask.ndim - 1) + (x_dim,))
    x_rel = x_rel / max(x_dim - 1, 1)
    if bool(class1_is_high_x):
        class1_side = np.broadcast_to(x_rel >= float(split_ratio), fg_mask.shape)
    else:
        class1_side = np.broadcast_to(x_rel < float(split_ratio), fg_mask.shape)
    class2_side = ~class1_side

    for cc_id in range(1, n_cc + 1):
        comp_mask = comp_arr == cc_id
        if not np.any(comp_mask):
            continue

        class1_part = comp_mask & class1_side
        class2_part = comp_mask & class2_side
        if int(class1_part.sum()) > 0 and int(class2_part.sum()) > 0:
            out[class1_part] = 1
            out[class2_part] = 2
            continue

        out[comp_mask] = 1 if int(class1_part.sum()) >= int(class2_part.sum()) else 2

    return out


def maybe_apply_side_relabel_postprocess(pred: np.ndarray, args) -> np.ndarray:
    if bool(getattr(args, "disable_side_relabel_postprocess", False)):
        return pred
    return relabel_foreground_by_geometry(
        pred=pred,
        class1_is_high_x=bool(getattr(args, "class1_is_high_x", True)),
        split_ratio=float(args.side_relabel_split),
    )


def summarize_case_metrics(pred: np.ndarray, gt: np.ndarray, spacing_mm: Sequence[float]) -> Dict[str, float]:
    pred_c1 = pred == 1
    pred_c2 = pred == 2
    pred_fg = pred > 0
    gt_c1 = gt == 1
    gt_c2 = gt == 2
    gt_fg = gt > 0

    d1 = binary_dice(pred_c1, gt_c1)
    d2 = binary_dice(pred_c2, gt_c2)
    d_fg = binary_dice(pred_fg, gt_fg)
    d1_swap = binary_dice(pred_c1, gt_c2)
    d2_swap = binary_dice(pred_c2, gt_c1)
    d_mean = float((d1 + d2) * 0.5)
    d_mean_swap = float((d1_swap + d2_swap) * 0.5)
    d_mean_oracle = max(d_mean, d_mean_swap)
    h1 = binary_hd95(pred_c1, gt_c1, spacing_mm)
    h2 = binary_hd95(pred_c2, gt_c2, spacing_mm)
    h_mean = float((h1 + h2) * 0.5)
    return {
        "dice_cls1": float(d1),
        "dice_cls2": float(d2),
        "dice_fg": float(d_fg),
        "dice_mean": float(d_mean),
        "dice_mean_oracle": float(d_mean_oracle),
        "swap_gap": float(d_mean_oracle - d_mean),
        "hd95_cls1": float(h1),
        "hd95_cls2": float(h2),
        "hd95_mean": float(h_mean),
    }


def evaluate_validation(
    model: torch.nn.Module,
    val_cases: List[CaseItem],
    args,
    device: torch.device,
    epoch_idx: int,
    use_amp: bool,
    criterion: Optional[DiceCELoss] = None,
    prior_eval_model: Optional[PriorNet] = None,
    pseudo_crop_transform=None,
) -> Dict[str, object]:
    model.eval()

    if len(val_cases) == 0:
        return {
            "epoch": int(epoch_idx),
            "num_cases": 0,
            "val_loss": float("nan"),
            "dice_cls1": float("nan"),
            "dice_cls2": float("nan"),
            "dice_fg": float("nan"),
            "dice_mean": float("nan"),
            "swap_gap": float("nan"),
            "pseudo_p_dice_cls1": float("nan"),
            "pseudo_p_dice_cls2": float("nan"),
            "pseudo_p_dice_fg": float("nan"),
            "pseudo_p_dice": float("nan"),
            "hd95_cls1": float("nan"),
            "hd95_cls2": float("nan"),
            "hd95_mean": float("nan"),
            "pred_fg_vox_mean": float("nan"),
            "empty_pred_rate": float("nan"),
        }

    sum_dice_cls1 = 0.0
    sum_dice_cls2 = 0.0
    sum_dice_fg = 0.0
    sum_dice_mean = 0.0
    sum_swap_gap = 0.0
    sum_val_loss = 0.0
    sum_pseudo_p_dice_cls1 = 0.0
    sum_pseudo_p_dice_cls2 = 0.0
    sum_pseudo_p_dice_fg = 0.0
    sum_pseudo_p_dice = 0.0
    sum_hd95_cls1 = 0.0
    sum_hd95_cls2 = 0.0
    sum_hd95_mean = 0.0
    sum_pred_fg_vox = 0.0
    sum_empty_pred = 0.0

    for case in tqdm(val_cases, desc=f"Val E{epoch_idx}", leave=True, dynamic_ncols=True, mininterval=1.5):
        pseudo_case, _ = generate_pseudo_case(
            model=model,
            case=case,
            args=args,
            device=device,
            epoch_idx=epoch_idx,
            use_amp=use_amp,
            criterion=criterion,
            prior_eval_model=prior_eval_model,
            use_prior_segment_for_eval=True,
            use_prior_segment_for_p=bool(args.use_prior_strong_p),
        )
        pred_raw = np.asarray(pseudo_case["vista_v_pred"], dtype=np.int16)

        gt, meta = priornet.load_volume(case.label)
        gt = np.asarray(gt, dtype=np.int16)
        if pred_raw.shape != gt.shape:
            raise ValueError(
                f"Shape mismatch in validation for {case.case_id}: pred={pred_raw.shape}, gt={gt.shape}"
            )

        spacing_mm = priornet.get_spacing_mm(meta)
        pred = maybe_apply_side_relabel_postprocess(pred_raw, args)
        pred_fg_vox = int((pred > 0).sum())
        metrics = summarize_case_metrics(pred, gt, spacing_mm)
        sum_val_loss += float(pseudo_case.get("vista_v_loss", float("nan")))
        sum_pred_fg_vox += float(pred_fg_vox)
        sum_empty_pred += float(pred_fg_vox <= 0)

        patch_pseudo_metrics = None
        if pseudo_crop_transform is not None:
            patch_pseudo_metrics = summarize_validation_patch_pseudo_case(
                model=model,
                case=case,
                args=args,
                device=device,
                epoch_idx=epoch_idx,
                use_amp=use_amp,
                prior_eval_model=prior_eval_model,
                crop_transform=pseudo_crop_transform,
            )
        if patch_pseudo_metrics is None:
            pseudo_p_label = np.asarray(pseudo_case["pseudo_p_label"], dtype=np.int16)
            pseudo_p_c1 = pseudo_p_label == 1
            pseudo_p_c2 = pseudo_p_label == 2
            pseudo_p_fg = pseudo_p_label > 0
            gt_c1 = gt == 1
            gt_c2 = gt == 2
            gt_fg = gt > 0
            pseudo_p_dice_cls1 = binary_dice(pseudo_p_c1, gt_c1)
            pseudo_p_dice_cls2 = binary_dice(pseudo_p_c2, gt_c2)
            pseudo_p_dice_fg = binary_dice(pseudo_p_fg, gt_fg)
        else:
            pseudo_p_dice_cls1 = float(patch_pseudo_metrics["dice_cls1"])
            pseudo_p_dice_cls2 = float(patch_pseudo_metrics["dice_cls2"])
            pseudo_p_dice_fg = float(patch_pseudo_metrics["dice_fg"])
        pseudo_p_dice = pseudo_p_dice_fg

        sum_dice_cls1 += float(metrics["dice_cls1"])
        sum_dice_cls2 += float(metrics["dice_cls2"])
        sum_dice_fg += float(metrics["dice_fg"])
        sum_dice_mean += float(metrics["dice_mean"])
        sum_swap_gap += float(metrics["swap_gap"])
        sum_pseudo_p_dice_cls1 += float(pseudo_p_dice_cls1)
        sum_pseudo_p_dice_cls2 += float(pseudo_p_dice_cls2)
        sum_pseudo_p_dice_fg += float(pseudo_p_dice_fg)
        sum_pseudo_p_dice += float(pseudo_p_dice)
        sum_hd95_cls1 += float(metrics["hd95_cls1"])
        sum_hd95_cls2 += float(metrics["hd95_cls2"])
        sum_hd95_mean += float(metrics["hd95_mean"])

    num_cases = max(1, len(val_cases))
    return {
        "epoch": int(epoch_idx),
        "num_cases": int(len(val_cases)),
        "val_loss": float(sum_val_loss / num_cases),
        "dice_cls1": float(sum_dice_cls1 / num_cases),
        "dice_cls2": float(sum_dice_cls2 / num_cases),
        "dice_fg": float(sum_dice_fg / num_cases),
        "dice_mean": float(sum_dice_mean / num_cases),
        "swap_gap": float(sum_swap_gap / num_cases),
        "pseudo_p_dice_cls1": float(sum_pseudo_p_dice_cls1 / num_cases),
        "pseudo_p_dice_cls2": float(sum_pseudo_p_dice_cls2 / num_cases),
        "pseudo_p_dice_fg": float(sum_pseudo_p_dice_fg / num_cases),
        "pseudo_p_dice": float(sum_pseudo_p_dice / num_cases),
        "hd95_cls1": float(sum_hd95_cls1 / num_cases),
        "hd95_cls2": float(sum_hd95_cls2 / num_cases),
        "hd95_mean": float(sum_hd95_mean / num_cases),
        "pred_fg_vox_mean": float(sum_pred_fg_vox / num_cases),
        "empty_pred_rate": float(sum_empty_pred / num_cases),
    }


def is_better_validation(
    dice_mean: float,
    hd95_mean: float,
    best_dice_mean: float,
    best_hd95_mean: float,
) -> bool:
    if not np.isfinite(dice_mean):
        return False
    if not np.isfinite(best_dice_mean):
        return True
    if dice_mean > best_dice_mean + 1e-6:
        return True
    if abs(dice_mean - best_dice_mean) <= 1e-6:
        if np.isfinite(hd95_mean) and not np.isfinite(best_hd95_mean):
            return True
        return hd95_mean < best_hd95_mean - 1e-6
    return False


def generate_pseudo_case(
    model: torch.nn.Module,
    case: CaseItem,
    args,
    device: torch.device,
    epoch_idx: int,
    use_amp: bool,
    criterion: Optional[DiceCELoss] = None,
    prior_eval_model: Optional[PriorNet] = None,
    use_prior_segment_for_eval: bool = False,
    use_prior_segment_for_p: bool = False,
    use_vista_guidance: bool = True,
    guide_with_gt: bool = False,
) -> Tuple[Dict[str, object], Dict[str, int]]:
    model.eval()

    with torch.no_grad():
        vol, meta = priornet.load_volume(case.image)
        vol = np.asarray(vol, dtype=np.float32)
        spacing_mm = priornet.get_spacing_mm(meta)
        # PriorNet operates directly in raw HU space. Intensity normalization is only
        # applied later for VISTA3D inference / patch training.
        prior_p_prob = priornet.compute_air_probability(
            vol,
            spacing_mm=spacing_mm,
            affine=meta.get("affine", None),
            exclude_midline=bool(args.exclude_midline),
            air_ref_method="otsu",
            component_prob_thr=float(args.prior_component_prob_thr),
            prob_gamma=float(args.prior_prob_gamma),
            enable_nonair_branch=True,
            nonair_weight=float(args.nonair_weight),
            nonair_seed_weight=float(args.nonair_seed_weight),
            nonair_center_q=float(args.nonair_center_q),
            nonair_sigma_hu=float(args.nonair_sigma_hu),
            nonair_bone_q=float(args.nonair_bone_q),
            nonair_gate_floor=float(args.nonair_gate_floor),
        ).astype(np.float32)
        image_norm = normalize_intensity_np(vol, args.norm_p_low, args.norm_p_high)
        global_x_map = build_global_x_coord_map(vol.shape)
        prior_p_prob_for_pseudo = prior_p_prob
        prior_p_fg_raw = np.asarray(prior_p_prob >= float(args.default_prob_thr), dtype=bool)
        prior_p_fg_strong = prior_p_fg_raw
        prior_p_fg_eval = prior_p_fg_raw
        if bool(use_prior_segment_for_eval) or bool(use_prior_segment_for_p):
            seg_model = prior_eval_model if prior_eval_model is not None else PriorNet(PriorNetConfig())
            seg_bin, _ = seg_model.segment_from_probability(
                prob=prior_p_prob,
                meta=meta,
                threshold=float(args.prior_p_thr),
            )
            prior_p_fg_strong = np.asarray(seg_bin > 0, dtype=bool)
            if bool(use_prior_segment_for_eval):
                prior_p_fg_eval = prior_p_fg_strong
            if bool(use_prior_segment_for_p):
                # Use PriorNet strong post-processing result as prior_p backbone.
                prior_p_prob_for_pseudo = np.where(
                    prior_p_fg_strong,
                    np.maximum(prior_p_prob, float(args.prior_p_thr)),
                    1e-5,
                ).astype(np.float32)

        if bool(guide_with_gt):
            if case.label is None:
                raise ValueError(f"GT-guided co-refinement requires label for case {case.case_id}.")
            gt_label, _ = priornet.load_volume(case.label)
            gt_label = np.asarray(gt_label, dtype=np.int16)
            if gt_label.shape != vol.shape:
                raise ValueError(
                    f"Shape mismatch in GT-guided co-refinement for {case.case_id}: "
                    f"image={vol.shape}, gt={gt_label.shape}"
                )
            gt_fg = np.asarray(gt_label > 0, dtype=bool)
            best_thr, _ = scan_best_threshold(
                prior_prob=prior_p_prob_for_pseudo,
                guidance_fg=gt_fg,
                sweep_start=float(args.sweep_start),
                sweep_end=float(args.sweep_end),
                sweep_step=float(args.sweep_step),
                fallback_thr=float(args.prior_p_thr if bool(use_prior_segment_for_p) else args.default_prob_thr),
                guidance_min_vox=1,
            )
            stats = {
                "trainable_cases": int(gt_fg.any()),
                "empty_cases": int(not gt_fg.any()),
                "skipped_small_cases": 0,
                "fallback_coarse_cases": 0,
                "fallback_seed_cases": 0,
                "cls2_cases": int(np.any(gt_label == 2)),
            }
            return {
                "image_norm": image_norm.astype(np.float32, copy=False),
                "vista_v_pred": None,
                "vista_v_fg_pred": None,
                "vista_v_loss": float("nan"),
                "prior_p_fg": np.asarray(prior_p_prob_for_pseudo >= float(best_thr), dtype=bool),
                "pseudo_p_fg": gt_fg,
                "pseudo_p_label": gt_label.astype(np.int16, copy=False),
                "used_for_train": bool(gt_fg.any()),
                "best_thr": float(best_thr),
            }, stats

        if not bool(use_vista_guidance):
            prior_only_fg = np.asarray(prior_p_fg_strong if bool(use_prior_segment_for_p) else prior_p_fg_raw, dtype=bool)
            pseudo_p_label = assign_pseudo_classes_from_prior(
                refined_fg=prior_only_fg,
                global_x_map=global_x_map,
                args=args,
                preserve_fg_boundary=bool(use_prior_segment_for_p),
            ).astype(np.int16, copy=False)
            pseudo_p_fg = np.asarray(pseudo_p_label > 0, dtype=bool)
            pseudo_vox = int(pseudo_p_fg.sum())
            cls2_vox = int((pseudo_p_label == 2).sum())
            used_for_train = pseudo_vox >= int(args.pseudo_train_min_vox)
            stats = {
                "trainable_cases": int(used_for_train),
                "empty_cases": int(pseudo_vox == 0),
                "skipped_small_cases": int(not used_for_train),
                "fallback_coarse_cases": 0,
                "fallback_seed_cases": 0,
                "cls2_cases": int(cls2_vox > 0),
            }
            return {
                "image_norm": image_norm.astype(np.float32, copy=False),
                "vista_v_pred": None,
                "vista_v_fg_pred": None,
                "vista_v_loss": float("nan"),
                "prior_p_fg": prior_p_fg_eval,
                "pseudo_p_fg": pseudo_p_fg,
                "pseudo_p_label": pseudo_p_label,
                "used_for_train": bool(used_for_train),
                "best_thr": float(args.prior_p_thr if bool(use_prior_segment_for_p) else args.default_prob_thr),
            }, stats

        # VISTA3D sees the normalized full volume here only to constrain the prior.
        # Augmentation is applied later, right before the unsupervised optimization step.
        image_t = torch.from_numpy(image_norm[None, None]).to(device)
        amp_ctx = cuda_autocast(enabled=True) if use_amp else nullcontext()
        with amp_ctx:
            logits = sliding_window_inference(
                image_t,
                roi_size=tuple(int(x) for x in args.roi_size),
                sw_batch_size=int(args.sw_batch),
                predictor=model,
            )
        vista_v_loss = float("nan")
        if criterion is not None and case.label is not None:
            gt_eval, _ = priornet.load_volume(case.label)
            gt_eval = np.asarray(gt_eval, dtype=np.int16)
            if gt_eval.shape != vol.shape:
                raise ValueError(
                    f"Shape mismatch in validation loss for {case.case_id}: pred={vol.shape}, gt={gt_eval.shape}"
                )
            gt_eval_t = torch.from_numpy(gt_eval[None, None]).to(device=device, dtype=torch.long)
            vista_multiclass_logits, _ = split_vista_output_logits(logits)
            vista_v_loss = float(
                compute_corefined_loss(
                    criterion=criterion,
                    multiclass_logits=vista_multiclass_logits,
                    target=gt_eval_t,
                    context="validation loss evaluation",
                ).item()
            )
        vista_multiclass_probs_t, vista_fg_prob_t = vista_output_probs_from_logits(logits)
        vista_v_probs = torch.cat((vista_multiclass_probs_t, vista_fg_prob_t), dim=1)[0].detach().cpu().numpy().astype(np.float32)
        vista_v_pred = torch.argmax(vista_multiclass_probs_t, dim=1)[0].detach().cpu().numpy().astype(np.int16)
        vista_v_fg_pred = (vista_fg_prob_t[0, 0] >= 0.5).detach().cpu().numpy().astype(bool)

        guided_fg_prob, vista_fg_prob, fg_seed, bg_seed, guidance_fg = compute_guided_prior_prob(
            prior_prob=prior_p_prob_for_pseudo,
            vista_probs=vista_v_probs,
            args=args,
            epoch_idx=epoch_idx,
        )

        coarse_thr, _ = scan_best_threshold(
            prior_prob=guided_fg_prob,
            guidance_fg=guidance_fg,
            sweep_start=float(args.sweep_start),
            sweep_end=float(args.sweep_end),
            sweep_step=float(args.sweep_step),
            fallback_thr=float(args.default_prob_thr),
            guidance_min_vox=int(args.guidance_min_vox),
        )

        coarse_fg = guided_fg_prob >= float(coarse_thr)
        coarse_fg = select_structured_components(
            fg_mask=coarse_fg,
            refined_prob=guided_fg_prob,
            vista_probs=vista_v_probs,
            vista_fg_prob=vista_fg_prob,
            fg_seed=fg_seed,
            bg_seed=bg_seed,
            global_x_map=global_x_map,
            args=args,
            epoch_idx=epoch_idx,
        )

        roi_mask = np.asarray(
            coarse_fg | fg_seed | (guided_fg_prob >= float(args.crf_roi_prob_thr)),
            dtype=bool,
        )
        if roi_mask.any():
            roi_mask = ndi.binary_dilation(
                roi_mask,
                structure=ndi.generate_binary_structure(roi_mask.ndim, 1),
                iterations=1,
            )

        refined_prob = run_densecrf_on_roi(
            guided_fg_prob=guided_fg_prob,
            vista_probs=vista_v_probs,
            image_norm=image_norm,
            roi_mask=roi_mask,
            args=args,
        )

        best_thr, _ = scan_best_threshold(
            prior_prob=refined_prob,
            guidance_fg=guidance_fg,
            sweep_start=float(args.sweep_start),
            sweep_end=float(args.sweep_end),
            sweep_step=float(args.sweep_step),
            fallback_thr=float(coarse_thr),
            guidance_min_vox=int(args.guidance_min_vox),
        )

        refined_fg = refined_prob >= float(best_thr)
        refined_fg &= roi_mask
        refined_fg = select_structured_components(
            fg_mask=refined_fg,
            refined_prob=refined_prob,
            vista_probs=vista_v_probs,
            vista_fg_prob=vista_fg_prob,
            fg_seed=fg_seed,
            bg_seed=bg_seed,
            global_x_map=global_x_map,
            args=args,
            epoch_idx=epoch_idx,
        )
        refined_fg, fg_source = select_pseudo_foreground_mask(
            refined_fg=refined_fg,
            coarse_fg=coarse_fg,
            fg_seed=fg_seed,
            args=args,
        )
        pseudo_p_label = assign_pseudo_classes_from_vista(
            refined_fg=refined_fg,
            vista_probs=vista_v_probs,
            global_x_map=global_x_map,
            args=args,
            epoch_idx=epoch_idx,
            preserve_fg_boundary=False,
        ).astype(np.int16, copy=False)

        pseudo_p_fg = np.asarray(pseudo_p_label > 0, dtype=bool)
        pseudo_vox = int(pseudo_p_fg.sum())
        cls2_vox = int((pseudo_p_label == 2).sum())
        used_for_train = pseudo_vox >= int(args.pseudo_train_min_vox)

    stats = {
        "trainable_cases": int(used_for_train),
        "empty_cases": int(pseudo_vox == 0),
        "skipped_small_cases": int(not used_for_train),
        "fallback_coarse_cases": int(fg_source == "coarse"),
        "fallback_seed_cases": int(fg_source == "seed"),
        "cls2_cases": int(cls2_vox > 0),
    }
    return {
        "image_norm": image_norm.astype(np.float32, copy=False),
        "vista_v_pred": vista_v_pred,
        "vista_v_fg_pred": vista_v_fg_pred,
        "vista_v_loss": float(vista_v_loss),
        "prior_p_fg": prior_p_fg_eval,
        "pseudo_p_fg": pseudo_p_fg,
        "pseudo_p_label": pseudo_p_label,
        "used_for_train": bool(used_for_train),
        "best_thr": float(best_thr),
    }, stats


def safe_stat_mean(values: Sequence[object]) -> float:
    if len(values) <= 0:
        return float("nan")
    arr = np.asarray(values, dtype=np.float32)
    return float(np.mean(arr))


def safe_stat_median(values: Sequence[object]) -> float:
    if len(values) <= 0:
        return float("nan")
    arr = np.asarray(values, dtype=np.float32)
    return float(np.median(arr))


def init_epoch_patch_stats(source: str) -> Dict[str, object]:
    return {
        "source": str(source),
        "case_count": 0,
        "patch_candidates": 0,
        "patch_accepted": 0,
        "fg_vox_list": [],
        "cls1_vox_list": [],
        "cls2_vox_list": [],
        "empty_count": 0,
        "one_side_only_count": 0,
        "reject_min_vox_count": 0,
        "reject_max_vox_count": 0,
        "reject_min_cls_vox_count": 0,
        "reject_require_both_classes_count": 0,
    }


def merge_epoch_patch_stats(epoch_stats: Dict[str, object], case_stats: Dict[str, object]) -> None:
    epoch_stats["case_count"] = int(epoch_stats["case_count"]) + 1
    for key in (
        "patch_candidates",
        "patch_accepted",
        "empty_count",
        "one_side_only_count",
        "reject_min_vox_count",
        "reject_max_vox_count",
        "reject_min_cls_vox_count",
        "reject_require_both_classes_count",
    ):
        epoch_stats[key] = int(epoch_stats[key]) + int(case_stats.get(key, 0))
    for key in ("fg_vox_list", "cls1_vox_list", "cls2_vox_list"):
        dst = epoch_stats.get(key, [])
        src = case_stats.get(key, [])
        if isinstance(dst, list) and isinstance(src, list):
            dst.extend(int(v) for v in src)


def summarize_epoch_patch_stats(prefix: str, stats: Dict[str, object]) -> Dict[str, object]:
    patch_candidates = int(stats.get("patch_candidates", 0))
    patch_accepted = int(stats.get("patch_accepted", 0))
    fg_vox_list = stats.get("fg_vox_list", [])
    cls1_vox_list = stats.get("cls1_vox_list", [])
    cls2_vox_list = stats.get("cls2_vox_list", [])
    denom = float(patch_candidates) if patch_candidates > 0 else float("nan")
    return {
        f"{prefix}_case_count": int(stats.get("case_count", 0)),
        f"{prefix}_patch_candidates": int(patch_candidates),
        f"{prefix}_patch_accepted": int(patch_accepted),
        f"{prefix}_accept_rate": float(patch_accepted / denom) if patch_candidates > 0 else float("nan"),
        f"{prefix}_fg_vox_mean": safe_stat_mean(fg_vox_list if isinstance(fg_vox_list, list) else []),
        f"{prefix}_fg_vox_median": safe_stat_median(fg_vox_list if isinstance(fg_vox_list, list) else []),
        f"{prefix}_cls1_vox_mean": safe_stat_mean(cls1_vox_list if isinstance(cls1_vox_list, list) else []),
        f"{prefix}_cls2_vox_mean": safe_stat_mean(cls2_vox_list if isinstance(cls2_vox_list, list) else []),
        f"{prefix}_empty_rate": float(int(stats.get('empty_count', 0)) / denom) if patch_candidates > 0 else float("nan"),
        f"{prefix}_one_side_only_rate": float(int(stats.get('one_side_only_count', 0)) / denom)
        if patch_candidates > 0
        else float("nan"),
        f"{prefix}_reject_min_vox_count": int(stats.get("reject_min_vox_count", 0)),
        f"{prefix}_reject_max_vox_count": int(stats.get("reject_max_vox_count", 0)),
        f"{prefix}_reject_min_cls_vox_count": int(stats.get("reject_min_cls_vox_count", 0)),
        f"{prefix}_reject_require_both_classes_count": int(stats.get("reject_require_both_classes_count", 0)),
    }


def summarize_case_patch_stats(epoch: int, case_stats: Dict[str, object]) -> Dict[str, object]:
    patch_candidates = int(case_stats.get("patch_candidates", 0))
    patch_accepted = int(case_stats.get("patch_accepted", 0))
    fg_vox_list = case_stats.get("fg_vox_list", [])
    cls1_vox_list = case_stats.get("cls1_vox_list", [])
    cls2_vox_list = case_stats.get("cls2_vox_list", [])
    denom = float(patch_candidates) if patch_candidates > 0 else float("nan")
    return {
        "epoch": int(epoch),
        "case_id": str(case_stats.get("case_id", "")),
        "source": str(case_stats.get("source", "")),
        "case_patch_candidates": int(patch_candidates),
        "case_patch_accepted": int(patch_accepted),
        "case_accept_rate": float(patch_accepted / denom) if patch_candidates > 0 else float("nan"),
        "case_fg_vox_mean": safe_stat_mean(fg_vox_list if isinstance(fg_vox_list, list) else []),
        "case_fg_vox_median": safe_stat_median(fg_vox_list if isinstance(fg_vox_list, list) else []),
        "case_cls1_vox_mean": safe_stat_mean(cls1_vox_list if isinstance(cls1_vox_list, list) else []),
        "case_cls2_vox_mean": safe_stat_mean(cls2_vox_list if isinstance(cls2_vox_list, list) else []),
        "case_empty_rate": float(int(case_stats.get("empty_count", 0)) / denom) if patch_candidates > 0 else float("nan"),
        "case_one_side_only_rate": float(int(case_stats.get("one_side_only_count", 0)) / denom)
        if patch_candidates > 0
        else float("nan"),
        "case_reject_min_vox_count": int(case_stats.get("reject_min_vox_count", 0)),
        "case_reject_max_vox_count": int(case_stats.get("reject_max_vox_count", 0)),
        "case_reject_min_cls_vox_count": int(case_stats.get("reject_min_cls_vox_count", 0)),
        "case_reject_require_both_classes_count": int(case_stats.get("reject_require_both_classes_count", 0)),
    }


def collate_patch_samples(samples: List[Dict[str, object]], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    images = torch.stack([torch.as_tensor(sample["image"]).float() for sample in samples], dim=0).to(device)
    labels = torch.stack([torch.as_tensor(sample["label"]).long() for sample in samples], dim=0).to(device)
    if labels.ndim == 4:
        labels = labels.unsqueeze(1)
    return images, labels


def backward_corefined_chunk(
    model: torch.nn.Module,
    samples: List[Dict[str, object]],
    criterion: DiceCELoss,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    loss_weight: float,
    accum_steps: int,
    step_idx: int,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float]:
    model.train()
    img_u, lbl_u = collate_patch_samples(samples, device)
    amp_ctx = cuda_autocast(enabled=True) if use_amp else nullcontext()
    with amp_ctx:
        logit_u = model(img_u)
        multiclass_logits, _ = split_vista_output_logits(logit_u)
        loss_value = compute_corefined_loss(
            criterion=criterion,
            multiclass_logits=multiclass_logits,
            target=lbl_u,
            context="co-refined training",
        )
        loss = (float(loss_weight) * loss_value) / float(accum_steps)
    scaler.scale(loss).backward()
    if step_idx % accum_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    loss_value_item = float(loss_value.item())
    return float(loss_value_item), float(loss_weight * loss_value_item)


def consume_corefined_buffer(
    patch_buffer: List[Dict[str, object]],
    micro_batch: int,
    model: torch.nn.Module,
    criterion: DiceCELoss,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    loss_weight: float,
    accum_steps: int,
    next_step_idx: int,
    device: torch.device,
    use_amp: bool,
    pseudo_vram_keeper: PseudoVRAMKeeper,
    flush_all: bool = False,
) -> Tuple[float, float, int, int]:
    raw_loss_sum = 0.0
    weighted_loss_sum = 0.0
    num_steps = 0
    while len(patch_buffer) >= micro_batch or (flush_all and len(patch_buffer) > 0):
        chunk_size = min(len(patch_buffer), micro_batch)
        chunk = patch_buffer[:chunk_size]
        del patch_buffer[:chunk_size]
        next_step_idx += 1
        num_steps += 1
        pseudo_vram_keeper.release()
        raw_loss_value, weighted_loss_value = backward_corefined_chunk(
            model=model,
            samples=chunk,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            loss_weight=loss_weight,
            accum_steps=accum_steps,
            step_idx=next_step_idx,
            device=device,
            use_amp=use_amp,
        )
        raw_loss_sum += float(raw_loss_value)
        weighted_loss_sum += float(weighted_loss_value)
        pseudo_vram_keeper.observe()
        pseudo_vram_keeper.hold()
    return float(raw_loss_sum), float(weighted_loss_sum), int(num_steps), int(next_step_idx)


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    args,
    extra: Optional[Dict[str, object]] = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "loss": float(loss),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "args": vars(args),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def sanitize_log_value(value: object) -> object:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path: str, payload: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def init_training_logs(
    config_path: str,
    history_jsonl_path: str,
    history_csv_path: str,
    config_payload: Dict[str, object],
    fieldnames: Sequence[str],
) -> None:
    write_json(config_path, config_payload)
    os.makedirs(os.path.dirname(history_jsonl_path) or ".", exist_ok=True)
    with open(history_jsonl_path, "w", encoding="utf-8"):
        pass
    with open(history_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()


def append_training_log(
    history_jsonl_path: str,
    history_csv_path: str,
    fieldnames: Sequence[str],
    record: Dict[str, object],
) -> None:
    sanitized = {str(k): sanitize_log_value(v) for k, v in record.items()}
    with open(history_jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(sanitized, ensure_ascii=False) + "\n")
    row = {name: sanitized.get(name, None) for name in fieldnames}
    with open(history_csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writerow(row)


def load_pretrained_weights(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> Dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = checkpoint.get("model_state", None)
    if model_state is None:
        raise KeyError(f"Checkpoint missing `model_state`: {checkpoint_path}")
    model.load_state_dict(model_state, strict=True)
    return checkpoint


def train_corefined_stage(args) -> None:
    configure_torch_runtime(bool(args.deterministic))
    set_seed(int(args.seed))
    print(
        "Runtime determinism: "
        f"{'enabled' if bool(args.deterministic) else 'disabled'} "
        f"(cudnn_benchmark={torch.backends.cudnn.benchmark}, "
        f"cudnn_deterministic={torch.backends.cudnn.deterministic})"
    )

    dataset_root = os.path.abspath(args.dataset_root)
    split_json = args.split_json or os.path.join(dataset_root, "splits.json")
    val_every = max(1, int(args.val_every))

    labeled_cases, unlabeled_cases = load_split_cases(dataset_root, split_json, args.split_ratio)
    val_cases = load_validation_cases(dataset_root, split_json, args.split_ratio, args.val_key)
    print(
        f"dataset={dataset_root} split={args.split_ratio}: "
        f"labeled={len(labeled_cases)}, unlabeled={len(unlabeled_cases)}, "
        f"val({args.val_key})={len(val_cases)}"
    )
    orientation_cases = list({case.case_id: case for case in [*labeled_cases, *unlabeled_cases, *val_cases]}.values())
    orientation_summary = summarize_orientation_consistency(orientation_cases, lateral_axis=-1)
    args.class1_is_high_x = bool(orientation_summary["increasing_toward"] == "left")
    print(
        "Orientation check: "
        f"cases={int(orientation_summary['num_cases'])}, "
        f"array_axis={int(orientation_summary['array_axis'])}, "
        f"image_axis={int(orientation_summary['image_axis'])}, "
        f"increasing_toward={orientation_summary['increasing_toward']}, "
        f"class1_is_high_x={bool(args.class1_is_high_x)}, "
        f"direction={orientation_summary['direction']}"
    )
    if int(orientation_summary["label_header_mismatch_count"]) > 0:
        print(
            "Warning: image/label header mismatch detected for "
            f"{int(orientation_summary['label_header_mismatch_count'])} cases. "
            "Training will continue because this pipeline uses voxel-index alignment, not world-coordinate resampling. "
            f"Examples: {orientation_summary['label_header_mismatch_cases'][:8]}"
        )

    run_name = args.run_name or args.split_ratio.replace("%", "pct")
    run_dir = os.path.join(os.path.abspath(args.output_dir), run_name)
    os.makedirs(run_dir, exist_ok=True)

    best_ckpt = os.path.join(run_dir, "best_model.pth")
    last_ckpt = os.path.join(run_dir, "last_model.pth")
    metrics_tag = (str(args.val_key).strip() or "validation").replace(os.sep, "_").replace("/", "_")
    best_metrics_json = os.path.join(run_dir, f"best_{metrics_tag}_metrics.json")
    train_config_json = os.path.join(run_dir, "train_config.json")
    train_history_jsonl = os.path.join(run_dir, "train_history.jsonl")
    train_history_csv = os.path.join(run_dir, "train_history.csv")
    corefined_case_config_json = os.path.join(run_dir, "corefined_case_log_config.json")
    corefined_case_history_jsonl = os.path.join(run_dir, "corefined_case_history.jsonl")
    corefined_case_history_csv = os.path.join(run_dir, "corefined_case_history.csv")

    if torch.cuda.is_available():
        gpu_idx = parse_gpu_index(args.gpu)
        gpu_count = torch.cuda.device_count()
        if gpu_idx >= gpu_count:
            raise ValueError(f"--gpu={args.gpu} is out of range. Available GPU indices: 0..{gpu_count - 1}")
        torch.cuda.set_device(gpu_idx)
        device = torch.device(f"cuda:{gpu_idx}")
        print(f"Using device: cuda:{gpu_idx}")
    else:
        device = torch.device("cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    pseudo_vram_keeper = PseudoVRAMKeeper(
        device=device,
        enabled=bool(args.hold_vram_during_pseudo),
        chunk_mb=int(args.pseudo_vram_chunk_mb),
    )

    model = build_aorta_vista3d_model(in_channels=1, class_ids=(1, 2)).to(device)
    prior_eval_model = PriorNet(PriorNetConfig())
    if not args.init_checkpoint:
        raise ValueError("--init-checkpoint is required for --stage corefined.")
    pretrained_ckpt = ensure_exists(os.path.abspath(args.init_checkpoint), "init checkpoint")
    pretrained_meta = load_pretrained_weights(model, pretrained_ckpt, device=device)
    print(
        f"Loaded supervised pretrained weights: {pretrained_ckpt} "
        f"(epoch={int(pretrained_meta.get('epoch', -1))}, "
        f"split_ratio={pretrained_meta.get('args', {}).get('split_ratio', 'unknown')})"
    )
    optimizer = AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = make_cuda_grad_scaler(enabled=use_amp)

    corefined_criterion = DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=1.0)

    corefined_crop_tf = build_array_guided_crop_transform(
        roi_size=args.roi_size,
        num_samples=int(args.num_samples),
        prior_mask_key="prior_mask",
        extra_keys=("prior_prob", "global_x", "gt_label"),
    )
    val_pseudo_crop_tf = build_array_guided_crop_transform(
        roi_size=args.roi_size,
        num_samples=int(args.num_samples),
        prior_mask_key="prior_mask",
        extra_keys=("prior_prob", "global_x", "gt_label"),
    )

    history_fields = [
        "epoch",
        "epoch_seconds",
        "lr",
        "corefined_weight",
        "labeled_loss_weight",
        "unlabeled_loss_weight",
        "corefined_accum_steps",
        "pseudo_fg_reliance",
        "pseudo_class_reliance",
        "num_labeled_cases",
        "num_unlabeled_cases",
        "active_labeled_cases",
        "active_unlabeled_cases",
        "corefined_train_steps",
        "labeled_train_steps",
        "unlabeled_train_steps",
        "train_loss",
        "corefined_loss",
        "total_loss",
        "labeled_loss_mean",
        "unlabeled_loss_mean",
        "labeled_case_count",
        "labeled_patch_candidates",
        "labeled_patch_accepted",
        "labeled_accept_rate",
        "labeled_fg_vox_mean",
        "labeled_fg_vox_median",
        "labeled_cls1_vox_mean",
        "labeled_cls2_vox_mean",
        "labeled_empty_rate",
        "labeled_one_side_only_rate",
        "unlabeled_case_count",
        "unlabeled_patch_candidates",
        "unlabeled_patch_accepted",
        "unlabeled_accept_rate",
        "unlabeled_fg_vox_mean",
        "unlabeled_fg_vox_median",
        "unlabeled_cls1_vox_mean",
        "unlabeled_cls2_vox_mean",
        "unlabeled_empty_rate",
        "unlabeled_one_side_only_rate",
        "unlabeled_reject_min_vox_count",
        "unlabeled_reject_max_vox_count",
        "unlabeled_reject_min_cls_vox_count",
        "unlabeled_reject_require_both_classes_count",
        "val_num_cases",
        "val_loss",
        "val_dice_cls1",
        "val_dice_cls2",
        "val_dice_fg",
        "val_dice_mean",
        "val_swap_gap",
        "val_pred_fg_vox_mean",
        "val_empty_pred_rate",
        "val_pseudo_p_dice_cls1",
        "val_pseudo_p_dice_cls2",
        "val_pseudo_p_dice_fg",
        "val_pseudo_p_dice",
        "val_hd95_cls1",
        "val_hd95_cls2",
        "val_hd95_mean",
        "best_val_dice_mean",
        "best_val_hd95_mean",
    ]
    init_training_logs(
        config_path=train_config_json,
        history_jsonl_path=train_history_jsonl,
        history_csv_path=train_history_csv,
        config_payload={
            "args": vars(args),
            "run_dir": run_dir,
            "device": str(device),
            "pretrained_checkpoint": pretrained_ckpt,
            "pretrained_epoch": int(pretrained_meta.get("epoch", -1)),
            "dataset_root": dataset_root,
            "split_json": split_json,
            "split_ratio": str(args.split_ratio),
            "num_labeled_cases": int(len(labeled_cases)),
            "num_unlabeled_cases": int(len(unlabeled_cases)),
            "num_val_cases": int(len(val_cases)),
        },
        fieldnames=history_fields,
    )
    case_history_fields = [
        "epoch",
        "case_id",
        "source",
        "case_patch_candidates",
        "case_patch_accepted",
        "case_accept_rate",
        "case_fg_vox_mean",
        "case_fg_vox_median",
        "case_cls1_vox_mean",
        "case_cls2_vox_mean",
        "case_empty_rate",
        "case_one_side_only_rate",
        "case_reject_min_vox_count",
        "case_reject_max_vox_count",
        "case_reject_min_cls_vox_count",
        "case_reject_require_both_classes_count",
    ]
    init_training_logs(
        config_path=corefined_case_config_json,
        history_jsonl_path=corefined_case_history_jsonl,
        history_csv_path=corefined_case_history_csv,
        config_payload={
            "args": vars(args),
            "run_dir": run_dir,
            "fields": case_history_fields,
        },
        fieldnames=case_history_fields,
    )
    print(
        f"Training monitor logs: {train_history_jsonl} | {train_history_csv} "
        f"| {corefined_case_history_jsonl} | {corefined_case_history_csv}"
    )

    unlabeled_schedule = (
        "disabled"
        if bool(args.gt_only)
        else f"epoch {max(1, int(args.unlabeled_start_epoch)):03d}"
    )
    print(
        "schedule: sup-pretrained init, labeled GT-guided co-refinement from epoch 001, "
        f"unlabeled pseudo-guided co-refinement from {unlabeled_schedule}"
    )

    best_val_dice = float("nan")
    best_val_hd95 = float("inf")
    best_eval_summary: Optional[Dict[str, object]] = None

    for epoch in range(1, int(args.max_epochs) + 1):
        epoch_start_time = time.time()
        current_lr = get_current_lr(epoch, args)
        for param_group in optimizer.param_groups:
            param_group["lr"] = float(current_lr)
        current_corefined_weight = get_current_corefined_weight(epoch, args)
        labeled_loss_weight = get_source_corefined_weight("labeled", args)
        unlabeled_loss_weight = get_source_corefined_weight("unlabeled", args)
        epoch_corefined_loss = 0.0
        epoch_total_loss = 0.0
        corefined_train_steps = 0
        source_train_steps = {"labeled": 0, "unlabeled": 0}
        source_raw_loss_sum = {"labeled": 0.0, "unlabeled": 0.0}
        source_weighted_loss_sum = {"labeled": 0.0, "unlabeled": 0.0}
        source_patch_stats = {
            "labeled": init_epoch_patch_stats("labeled"),
            "unlabeled": init_epoch_patch_stats("unlabeled"),
        }
        active_labeled_cases = list(labeled_cases)
        active_unlabeled_cases = list(unlabeled_cases) if use_unlabeled_corefined(epoch, args) else []

        need_pseudo = (
            (len(active_unlabeled_cases) > 0 or len(active_labeled_cases) > 0)
            and epoch >= int(args.pseudo_start_epoch)
            and (epoch % int(args.pseudo_update_every) == 0)
        )
        pseudo_use_vista_guidance = True
        corefined_accum_steps = get_unsup_accum_steps(
            num_labeled=len(active_labeled_cases),
            num_unlabeled=len(active_unlabeled_cases),
            args=args,
        )
        if need_pseudo:
            print(f"[Epoch {epoch:03d}] Co-refined phase")
            if pseudo_vram_keeper.enabled and pseudo_vram_keeper.target_reserved_bytes > 0:
                pseudo_vram_keeper.hold()
            try:
                corefined_micro_batch = max(1, int(args.batchsize) * max(1, int(args.num_samples)))
                patch_buffers: Dict[str, List[Dict[str, object]]] = {"labeled": [], "unlabeled": []}
                co_refined_items: List[Tuple[CaseItem, str]] = (
                    [(case, "unlabeled") for case in active_unlabeled_cases]
                    + [(case, "labeled") for case in active_labeled_cases]
                )
                random.shuffle(co_refined_items)

                unsup_pbar = tqdm(
                    co_refined_items,
                    desc=f"CoRef E{epoch}/{args.max_epochs}",
                    leave=True,
                    dynamic_ncols=True,
                    mininterval=1.5,
                )
                optimizer.zero_grad(set_to_none=True)
                for case, source in unsup_pbar:
                    guide_with_gt = bool(source == "labeled")
                    patch_samples, case_stats = run_with_pseudo_vram_retry(
                        lambda: build_corefined_training_samples(
                            model=model,
                            case=case,
                            args=args,
                            device=device,
                            epoch_idx=epoch,
                            use_amp=use_amp,
                            prior_eval_model=prior_eval_model,
                            use_prior_segment_for_p=bool(args.use_prior_strong_p),
                            crop_transform=corefined_crop_tf,
                            use_vista_guidance=bool(pseudo_use_vista_guidance and not guide_with_gt),
                            guide_with_gt=bool(guide_with_gt),
                        ),
                        pseudo_vram_keeper,
                        desc=f"co-refined patch preparation for {case.case_id}",
                    )
                    merge_epoch_patch_stats(source_patch_stats[source], case_stats)
                    append_training_log(
                        history_jsonl_path=corefined_case_history_jsonl,
                        history_csv_path=corefined_case_history_csv,
                        fieldnames=case_history_fields,
                        record=summarize_case_patch_stats(epoch, case_stats),
                    )
                    pseudo_vram_keeper.hold()

                    total_source_weight = float(current_corefined_weight) * float(get_source_corefined_weight(source, args))
                    if len(patch_samples) > 0 and total_source_weight > 0.0:
                        patch_buffers[source].extend(patch_samples)

                        raw_loss_sum, weighted_loss_sum, num_steps, corefined_train_steps = consume_corefined_buffer(
                            patch_buffer=patch_buffers[source],
                            micro_batch=corefined_micro_batch,
                            model=model,
                            criterion=corefined_criterion,
                            optimizer=optimizer,
                            scaler=scaler,
                            loss_weight=total_source_weight,
                            accum_steps=corefined_accum_steps,
                            next_step_idx=corefined_train_steps,
                            device=device,
                            use_amp=use_amp,
                            pseudo_vram_keeper=pseudo_vram_keeper,
                        )
                        source_raw_loss_sum[source] += float(raw_loss_sum)
                        source_weighted_loss_sum[source] += float(weighted_loss_sum)
                        source_train_steps[source] += int(num_steps)

                for source in ("labeled", "unlabeled"):
                    total_source_weight = float(current_corefined_weight) * float(get_source_corefined_weight(source, args))
                    raw_loss_sum, weighted_loss_sum, num_steps, corefined_train_steps = consume_corefined_buffer(
                        patch_buffer=patch_buffers[source],
                        micro_batch=corefined_micro_batch,
                        model=model,
                        criterion=corefined_criterion,
                        optimizer=optimizer,
                        scaler=scaler,
                        loss_weight=total_source_weight,
                        accum_steps=corefined_accum_steps,
                        next_step_idx=corefined_train_steps,
                        device=device,
                        use_amp=use_amp,
                        pseudo_vram_keeper=pseudo_vram_keeper,
                        flush_all=True,
                    )
                    source_raw_loss_sum[source] += float(raw_loss_sum)
                    source_weighted_loss_sum[source] += float(weighted_loss_sum)
                    source_train_steps[source] += int(num_steps)

                if (
                    (float(current_corefined_weight) * max(float(labeled_loss_weight), float(unlabeled_loss_weight))) > 0.0
                    and corefined_train_steps > 0
                    and (corefined_train_steps % corefined_accum_steps != 0)
                ):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                if corefined_train_steps > 0:
                    epoch_corefined_loss = float(
                        (float(source_weighted_loss_sum["labeled"]) + float(source_weighted_loss_sum["unlabeled"]))
                        / float(corefined_train_steps)
                    )
                    epoch_total_loss = float(epoch_corefined_loss)
                    pseudo_vram_keeper.set_target_from_observed(headroom_mb=float(args.pseudo_vram_headroom_mb))
            finally:
                pseudo_vram_keeper.release()

        labeled_loss_mean = (
            float(source_raw_loss_sum["labeled"] / float(source_train_steps["labeled"]))
            if int(source_train_steps["labeled"]) > 0
            else float("nan")
        )
        unlabeled_loss_mean = (
            float(source_raw_loss_sum["unlabeled"] / float(source_train_steps["unlabeled"]))
            if int(source_train_steps["unlabeled"]) > 0
            else float("nan")
        )
        epoch_seconds = float(time.time() - epoch_start_time)

        val_summary: Optional[Dict[str, object]] = None
        if len(val_cases) > 0 and (epoch % val_every == 0):
            print(f"[Epoch {epoch:03d}] Validation phase")
            val_summary = evaluate_validation(
                model=model,
                val_cases=val_cases,
                args=args,
                device=device,
                epoch_idx=epoch,
                use_amp=use_amp,
                criterion=corefined_criterion,
                prior_eval_model=prior_eval_model,
                pseudo_crop_transform=val_pseudo_crop_tf,
            )

        if val_summary is not None:
            cur_dice = float(val_summary["dice_mean"])
            cur_hd95 = float(val_summary["hd95_mean"])
            if is_better_validation(cur_dice, cur_hd95, best_val_dice, best_val_hd95):
                best_val_dice = cur_dice
                best_val_hd95 = cur_hd95
                best_eval_summary = dict(val_summary)
                ckpt_extra = {
                    "eval_split": str(args.val_key),
                    "val_num_cases": int(val_summary["num_cases"]),
                    "val_loss": float(val_summary["val_loss"]),
                    "val_dice_cls1": float(val_summary["dice_cls1"]),
                    "val_dice_cls2": float(val_summary["dice_cls2"]),
                    "val_dice_fg": float(val_summary["dice_fg"]),
                    "val_dice_mean": float(val_summary["dice_mean"]),
                    "val_swap_gap": float(val_summary["swap_gap"]),
                    "val_pred_fg_vox_mean": float(val_summary["pred_fg_vox_mean"]),
                    "val_empty_pred_rate": float(val_summary["empty_pred_rate"]),
                    "val_pseudo_p_dice_cls1": float(val_summary["pseudo_p_dice_cls1"]),
                    "val_pseudo_p_dice_cls2": float(val_summary["pseudo_p_dice_cls2"]),
                    "val_pseudo_p_dice_fg": float(val_summary["pseudo_p_dice_fg"]),
                    "val_pseudo_p_dice": float(val_summary["pseudo_p_dice"]),
                    "val_hd95_cls1": float(val_summary["hd95_cls1"]),
                    "val_hd95_cls2": float(val_summary["hd95_cls2"]),
                    "val_hd95_mean": float(val_summary["hd95_mean"]),
                }
                save_checkpoint(best_ckpt, model, optimizer, epoch, epoch_total_loss, args, extra=ckpt_extra)
                os.makedirs(os.path.dirname(best_metrics_json) or ".", exist_ok=True)
                metrics_payload = {
                    "checkpoint": os.path.basename(best_ckpt),
                    "eval_split": str(args.val_key),
                    "epoch": int(val_summary["epoch"]),
                    "num_cases": int(val_summary["num_cases"]),
                    "dice_cls1": float(val_summary["dice_cls1"]),
                    "dice_cls2": float(val_summary["dice_cls2"]),
                    "dice_fg": float(val_summary["dice_fg"]),
                    "dice_mean": float(val_summary["dice_mean"]),
                    "swap_gap": float(val_summary["swap_gap"]),
                    "pred_fg_vox_mean": float(val_summary["pred_fg_vox_mean"]),
                    "empty_pred_rate": float(val_summary["empty_pred_rate"]),
                    "pseudo_p_dice_cls1": float(val_summary["pseudo_p_dice_cls1"]),
                    "pseudo_p_dice_cls2": float(val_summary["pseudo_p_dice_cls2"]),
                    "pseudo_p_dice_fg": float(val_summary["pseudo_p_dice_fg"]),
                    "pseudo_p_dice": float(val_summary["pseudo_p_dice"]),
                    "hd95_cls1": float(val_summary["hd95_cls1"]),
                    "hd95_cls2": float(val_summary["hd95_cls2"]),
                    "hd95_mean": float(val_summary["hd95_mean"]),
                    "selection_rule": "higher dice_mean, tie by lower hd95_mean",
                }
                for key, value in list(metrics_payload.items()):
                    if isinstance(value, (float, np.floating)):
                        metrics_payload[key] = float(value) if np.isfinite(value) else None
                    elif isinstance(value, (int, np.integer)):
                        metrics_payload[key] = int(value)
                with open(best_metrics_json, "w", encoding="utf-8") as f:
                    json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

        epoch_record: Dict[str, object] = {
            "epoch": int(epoch),
            "epoch_seconds": float(epoch_seconds),
            "lr": float(current_lr),
            "corefined_weight": float(current_corefined_weight),
            "labeled_loss_weight": float(labeled_loss_weight),
            "unlabeled_loss_weight": float(unlabeled_loss_weight),
            "corefined_accum_steps": int(corefined_accum_steps),
            "pseudo_fg_reliance": float(get_pseudo_fg_reliance(epoch, args)),
            "pseudo_class_reliance": float(get_pseudo_class_reliance(epoch, args)),
            "num_labeled_cases": int(len(labeled_cases)),
            "num_unlabeled_cases": int(len(unlabeled_cases)),
            "active_labeled_cases": int(len(active_labeled_cases)),
            "active_unlabeled_cases": int(len(active_unlabeled_cases)),
            "corefined_train_steps": int(corefined_train_steps),
            "labeled_train_steps": int(source_train_steps["labeled"]),
            "unlabeled_train_steps": int(source_train_steps["unlabeled"]),
            "train_loss": float(epoch_total_loss),
            "corefined_loss": float(epoch_corefined_loss),
            "total_loss": float(epoch_total_loss),
            "labeled_loss_mean": float(labeled_loss_mean),
            "unlabeled_loss_mean": float(unlabeled_loss_mean),
            "best_val_dice_mean": float(best_val_dice),
            "best_val_hd95_mean": float(best_val_hd95),
        }
        epoch_record.update(summarize_epoch_patch_stats("labeled", source_patch_stats["labeled"]))
        epoch_record.update(summarize_epoch_patch_stats("unlabeled", source_patch_stats["unlabeled"]))
        if val_summary is not None:
            epoch_record.update(
                {
                    "val_num_cases": int(val_summary["num_cases"]),
                    "val_loss": float(val_summary["val_loss"]),
                    "val_dice_cls1": float(val_summary["dice_cls1"]),
                    "val_dice_cls2": float(val_summary["dice_cls2"]),
                    "val_dice_fg": float(val_summary["dice_fg"]),
                    "val_dice_mean": float(val_summary["dice_mean"]),
                    "val_swap_gap": float(val_summary["swap_gap"]),
                    "val_pred_fg_vox_mean": float(val_summary["pred_fg_vox_mean"]),
                    "val_empty_pred_rate": float(val_summary["empty_pred_rate"]),
                    "val_pseudo_p_dice_cls1": float(val_summary["pseudo_p_dice_cls1"]),
                    "val_pseudo_p_dice_cls2": float(val_summary["pseudo_p_dice_cls2"]),
                    "val_pseudo_p_dice_fg": float(val_summary["pseudo_p_dice_fg"]),
                    "val_pseudo_p_dice": float(val_summary["pseudo_p_dice"]),
                    "val_hd95_cls1": float(val_summary["hd95_cls1"]),
                    "val_hd95_cls2": float(val_summary["hd95_cls2"]),
                    "val_hd95_mean": float(val_summary["hd95_mean"]),
                }
            )
        append_training_log(
            history_jsonl_path=train_history_jsonl,
            history_csv_path=train_history_csv,
            fieldnames=history_fields,
            record=epoch_record,
        )

        if val_summary is not None:
            print(
                f"Epoch {epoch:03d} | train_loss={epoch_total_loss:.4f} "
                f"corefined={epoch_corefined_loss:.4f} "
                f"w={current_corefined_weight:.3f} lr={current_lr:.2e} "
                f"u_accum={corefined_accum_steps} "
                f"lab_loss={labeled_loss_mean:.4f} unl_loss={unlabeled_loss_mean:.4f} "
                f"u_acc_rate={float(epoch_record['unlabeled_accept_rate']):.3f} total={epoch_total_loss:.4f} "
                f"val_loss={float(val_summary['val_loss']):.4f} "
                f"| val_dice_cls1={float(val_summary['dice_cls1']):.4f} "
                f"val_dice_cls2={float(val_summary['dice_cls2']):.4f} "
                f"val_dice_fg={float(val_summary['dice_fg']):.4f} "
                f"val_dice_mean={float(val_summary['dice_mean']):.4f} "
                f"swap_gap={float(val_summary['swap_gap']):.4f} "
                f"val_empty_pred_rate={float(val_summary['empty_pred_rate']):.4f} "
                f"pseudo_p_dice_cls1={float(val_summary['pseudo_p_dice_cls1']):.4f} "
                f"pseudo_p_dice_cls2={float(val_summary['pseudo_p_dice_cls2']):.4f} "
                f"pseudo_p_dice_fg={float(val_summary['pseudo_p_dice_fg']):.4f} "
                f"val_hd95={float(val_summary['hd95_mean']):.4f}"
            )
        else:
            print(
                f"Epoch {epoch:03d} | train_loss={epoch_total_loss:.4f} "
                f"corefined={epoch_corefined_loss:.4f} "
                f"w={current_corefined_weight:.3f} lr={current_lr:.2e} "
                f"u_accum={corefined_accum_steps} "
                f"lab_loss={labeled_loss_mean:.4f} unl_loss={unlabeled_loss_mean:.4f} "
                f"u_acc_rate={float(epoch_record['unlabeled_accept_rate']):.3f} total={epoch_total_loss:.4f}"
            )

    if os.path.exists(last_ckpt):
        os.remove(last_ckpt)

    if np.isfinite(best_val_dice):
        print(
            f"Training finished. Best checkpoint: {best_ckpt} "
            f"(best val_dice_mean={best_val_dice:.4f}, val_hd95={best_val_hd95:.4f})"
        )
        if best_eval_summary is not None:
            print(
                f"Best {args.val_key} metrics: {best_metrics_json} "
                f"(epoch={int(best_eval_summary['epoch'])}, cases={int(best_eval_summary['num_cases'])})"
            )
    else:
        print(
            f"Training finished. Best checkpoint selection was skipped because "
            f"no validation evaluation was executed (val_key={args.val_key}, val_every={val_every})."
        )


def sanitize_run_name(name: Optional[str], default_name: str) -> str:
    run_name = str(name).strip() if name is not None else ""
    if run_name == "":
        return default_name
    return run_name.replace("%", "pct").replace(os.sep, "_").replace("/", "_")


def build_output_prefix(split_ratio: str, run_name: Optional[str], default_name: str) -> str:
    split_tag = split_ratio_to_tag(split_ratio)
    run_tag = sanitize_run_name(run_name, default_name)
    return run_tag if split_tag in run_tag.split("_") else f"{run_tag}_{split_tag}"


def get_current_unsup_weight(epoch: int, args) -> float:
    pseudo_start_epoch = max(1, int(args.pseudo_start_epoch))
    if epoch < pseudo_start_epoch:
        return 0.0
    return float(args.unsup_weight)


def backward_unsup_chunk(
    model: torch.nn.Module,
    samples: List[Dict[str, object]],
    unsup_criterion: DiceCELoss,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    current_unsup_weight: float,
    unsup_accum_steps: int,
    step_idx: int,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.train()
    img_u, lbl_u = collate_patch_samples(samples, device)
    amp_ctx = cuda_autocast(enabled=True) if use_amp else nullcontext()
    with amp_ctx:
        logit_u = model(img_u)
        multiclass_logits, _ = split_vista_output_logits(logit_u)
        loss_value = compute_corefined_loss(
            criterion=unsup_criterion,
            multiclass_logits=multiclass_logits,
            target=lbl_u,
            context="unsupervised training",
        )
        loss = (float(current_unsup_weight) * loss_value) / float(unsup_accum_steps)
    scaler.scale(loss).backward()
    if step_idx % unsup_accum_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    return float(current_unsup_weight * float(loss_value.item()))


def evaluate_unsup_validation(
    model: torch.nn.Module,
    val_cases,
    args,
    device: torch.device,
    epoch_idx: int,
    use_amp: bool,
    prior_eval_model: Optional[PriorNet] = None,
) -> Dict[str, object]:
    model.eval()
    if len(val_cases) == 0:
        return {
            "epoch": int(epoch_idx),
            "num_cases": 0,
            "dice_cls1": float("nan"),
            "dice_cls2": float("nan"),
            "dice_fg": float("nan"),
            "dice_mean": float("nan"),
            "prior_p_dice": float("nan"),
            "pseudo_p_dice_fg": float("nan"),
            "hd95_mean": float("nan"),
        }

    sum_dice_cls1 = 0.0
    sum_dice_cls2 = 0.0
    sum_dice_fg = 0.0
    sum_dice_mean = 0.0
    sum_prior_p_dice = 0.0
    sum_pseudo_p_dice_fg = 0.0
    sum_hd95_mean = 0.0

    for case in tqdm(val_cases, desc=f"Val E{epoch_idx}", leave=True, dynamic_ncols=True, mininterval=1.5):
        pseudo_case, _ = generate_pseudo_case(
            model=model,
            case=case,
            args=args,
            device=device,
            epoch_idx=epoch_idx,
            use_amp=use_amp,
            criterion=None,
            prior_eval_model=prior_eval_model,
            use_prior_segment_for_eval=True,
            use_prior_segment_for_p=bool(args.use_prior_strong_p),
            use_vista_guidance=True,
        )
        pred = np.asarray(pseudo_case["vista_v_pred"], dtype=np.int16)
        gt, meta = priornet.load_volume(case.label)
        gt = np.asarray(gt, dtype=np.int16)
        spacing_mm = priornet.get_spacing_mm(meta)
        metrics = summarize_case_metrics(pred, gt, spacing_mm)
        prior_p_fg = np.asarray(pseudo_case["prior_p_fg"], dtype=bool)
        pseudo_p_fg = np.asarray(pseudo_case["pseudo_p_label"], dtype=np.int16) > 0
        sum_dice_cls1 += float(metrics["dice_cls1"])
        sum_dice_cls2 += float(metrics["dice_cls2"])
        sum_dice_fg += float(metrics["dice_fg"])
        sum_dice_mean += float(metrics["dice_mean"])
        sum_prior_p_dice += float(binary_dice(prior_p_fg, gt > 0))
        sum_pseudo_p_dice_fg += float(binary_dice(pseudo_p_fg, gt > 0))
        sum_hd95_mean += float(metrics["hd95_mean"])

    num_cases = float(max(1, len(val_cases)))
    return {
        "epoch": int(epoch_idx),
        "num_cases": int(len(val_cases)),
        "dice_cls1": float(sum_dice_cls1 / num_cases),
        "dice_cls2": float(sum_dice_cls2 / num_cases),
        "dice_fg": float(sum_dice_fg / num_cases),
        "dice_mean": float(sum_dice_mean / num_cases),
        "prior_p_dice": float(sum_prior_p_dice / num_cases),
        "pseudo_p_dice_fg": float(sum_pseudo_p_dice_fg / num_cases),
        "hd95_mean": float(sum_hd95_mean / num_cases),
    }


def train_unsup_stage(args) -> None:
    configure_torch_runtime(bool(args.deterministic))
    set_seed(int(args.seed))
    patch_monai_max_seed()

    dataset_root = os.path.abspath(args.dataset_root)
    split_json = args.split_json or os.path.join(dataset_root, "splits.json")
    unlabeled_labeled, unlabeled_cases = load_split_cases(dataset_root, split_json, args.split_ratio)
    if len(unlabeled_labeled) > 0:
        print(f"[unsup] ignoring {len(unlabeled_labeled)} labeled cases and training only on unlabeled data.")
    val_cases = load_validation_cases(dataset_root, split_json, args.split_ratio, args.val_key)

    prefix = build_output_prefix(args.split_ratio, args.run_name, "unsup")
    run_dir = os.path.join(os.path.abspath(args.output_dir), prefix)
    os.makedirs(run_dir, exist_ok=True)
    best_ckpt = os.path.join(run_dir, "best_model.pth")
    best_metrics_json = os.path.join(run_dir, f"best_{args.val_key or 'validation'}_metrics.json")

    if torch.cuda.is_available():
        gpu_idx = parse_gpu_index(args.gpu)
        if gpu_idx >= torch.cuda.device_count():
            raise ValueError(f"--gpu={args.gpu} is out of range. Available GPU indices: 0..{torch.cuda.device_count() - 1}")
        torch.cuda.set_device(gpu_idx)
        device = torch.device(f"cuda:{gpu_idx}")
    else:
        device = torch.device("cpu")
    use_amp = bool(args.amp and device.type == "cuda")

    model = build_aorta_vista3d_model(in_channels=1, class_ids=(1, 2)).to(device)
    prior_eval_model = PriorNet(PriorNetConfig())
    optimizer = AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = make_cuda_grad_scaler(enabled=use_amp)
    unsup_criterion = DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=1.0)
    unsup_tf = build_array_train_transform(roi_size=args.roi_size, num_samples=int(args.num_samples))
    best_val_dice = float("nan")
    best_val_hd95 = float("inf")
    unsup_micro_batch = max(1, int(args.batchsize) * max(1, int(args.num_samples)))

    for epoch in range(1, int(args.max_epochs) + 1):
        current_lr = get_current_lr(epoch, args)
        for param_group in optimizer.param_groups:
            param_group["lr"] = float(current_lr)
        current_unsup_weight = get_current_unsup_weight(epoch, args)
        unsup_accum_steps = get_unsup_accum_steps(num_labeled=0, num_unlabeled=len(unlabeled_cases), args=args)
        use_vista_guidance = bool(args.pseudo_use_vista_guidance and epoch >= int(args.pseudo_vista_guidance_start_epoch))
        patch_buffer: List[Dict[str, object]] = []
        epoch_unsup_loss = 0.0
        unsup_train_steps = 0
        optimizer.zero_grad(set_to_none=True)

        for case in tqdm(unlabeled_cases, desc=f"Unsup E{epoch}/{args.max_epochs}", leave=True, dynamic_ncols=True, mininterval=1.5):
            pseudo_case, _ = generate_pseudo_case(
                model=model,
                case=case,
                args=args,
                device=device,
                epoch_idx=epoch,
                use_amp=use_amp,
                criterion=None,
                prior_eval_model=prior_eval_model,
                use_prior_segment_for_eval=False,
                use_prior_segment_for_p=bool(args.use_prior_strong_p),
                use_vista_guidance=bool(use_vista_guidance),
                guide_with_gt=False,
            )
            if bool(pseudo_case["used_for_train"]) and current_unsup_weight > 0.0:
                aug_samples = ensure_sample_list(
                    unsup_tf(
                        {
                            "image": pseudo_case["image_norm"],
                            "label": pseudo_case["pseudo_p_label"].astype(np.float32, copy=False),
                        }
                    )
                )
                for sample in aug_samples:
                    patch_buffer.append(
                        {
                            "image": torch.as_tensor(sample["image"]).clone().float(),
                            "label": clone_label_tensor(sample["label"]),
                        }
                    )
                while len(patch_buffer) >= unsup_micro_batch:
                    chunk = patch_buffer[:unsup_micro_batch]
                    del patch_buffer[:unsup_micro_batch]
                    unsup_train_steps += 1
                    epoch_unsup_loss += backward_unsup_chunk(
                        model=model,
                        samples=chunk,
                        unsup_criterion=unsup_criterion,
                        optimizer=optimizer,
                        scaler=scaler,
                        current_unsup_weight=current_unsup_weight,
                        unsup_accum_steps=unsup_accum_steps,
                        step_idx=unsup_train_steps,
                        device=device,
                        use_amp=use_amp,
                    )

        if len(patch_buffer) > 0 and current_unsup_weight > 0.0:
            unsup_train_steps += 1
            epoch_unsup_loss += backward_unsup_chunk(
                model=model,
                samples=patch_buffer,
                unsup_criterion=unsup_criterion,
                optimizer=optimizer,
                scaler=scaler,
                current_unsup_weight=current_unsup_weight,
                unsup_accum_steps=unsup_accum_steps,
                step_idx=unsup_train_steps,
                device=device,
                use_amp=use_amp,
            )
        if current_unsup_weight > 0.0 and unsup_train_steps > 0 and (unsup_train_steps % unsup_accum_steps != 0):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if unsup_train_steps > 0:
            epoch_unsup_loss /= float(unsup_train_steps)

        val_summary = None
        if len(val_cases) > 0 and (epoch % max(1, int(args.val_every)) == 0):
            val_summary = evaluate_unsup_validation(
                model=model,
                val_cases=val_cases,
                args=args,
                device=device,
                epoch_idx=epoch,
                use_amp=use_amp,
                prior_eval_model=prior_eval_model,
            )
            cur_dice = float(val_summary["dice_mean"])
            cur_hd95 = float(val_summary["hd95_mean"])
            if is_better_validation(cur_dice, cur_hd95, best_val_dice, best_val_hd95):
                best_val_dice = cur_dice
                best_val_hd95 = cur_hd95
                save_checkpoint(best_ckpt, model, optimizer, epoch, epoch_unsup_loss, args, extra=val_summary)
                write_json(
                    best_metrics_json,
                    {
                        "checkpoint": os.path.basename(best_ckpt),
                        "epoch": int(val_summary["epoch"]),
                        "num_cases": int(val_summary["num_cases"]),
                        "dice_cls1": float(val_summary["dice_cls1"]),
                        "dice_cls2": float(val_summary["dice_cls2"]),
                        "dice_fg": float(val_summary["dice_fg"]),
                        "dice_mean": float(val_summary["dice_mean"]),
                        "prior_p_dice": float(val_summary["prior_p_dice"]),
                        "pseudo_p_dice_fg": float(val_summary["pseudo_p_dice_fg"]),
                        "hd95_mean": float(val_summary["hd95_mean"]),
                    },
                )

        if val_summary is not None:
            print(
                f"Epoch {epoch:03d} | unsup={epoch_unsup_loss:.4f} "
                f"lr={current_lr:.2e} val_dice_fg={float(val_summary['dice_fg']):.4f} "
                f"prior_p_dice={float(val_summary['prior_p_dice']):.4f}"
            )
        else:
            print(f"Epoch {epoch:03d} | unsup={epoch_unsup_loss:.4f} lr={current_lr:.2e}")


def evaluate_supervised_validation(
    model: torch.nn.Module,
    val_cases,
    args,
    device: torch.device,
    epoch_idx: int,
    use_amp: bool,
) -> Dict[str, object]:
    model.eval()
    if len(val_cases) == 0:
        return {
            "epoch": int(epoch_idx),
            "num_cases": 0,
            "dice_cls1": float("nan"),
            "dice_cls2": float("nan"),
            "dice_fg": float("nan"),
            "dice_mean": float("nan"),
            "hd95_mean": float("nan"),
        }

    sum_dice_cls1 = 0.0
    sum_dice_cls2 = 0.0
    sum_dice_fg = 0.0
    sum_dice_mean = 0.0
    sum_hd95_mean = 0.0
    model.eval()

    for case in tqdm(val_cases, desc=f"Val E{epoch_idx}", leave=True, dynamic_ncols=True, mininterval=1.5):
        vol, _ = priornet.load_volume(case.image)
        vol = np.asarray(vol, dtype=np.float32)
        image_norm = normalize_intensity_np(vol, args.norm_p_low, args.norm_p_high)
        image_t = torch.from_numpy(image_norm[None, None]).to(device)
        amp_ctx = cuda_autocast(enabled=True) if use_amp else nullcontext()
        with torch.no_grad():
            with amp_ctx:
                logits = sliding_window_inference(
                    image_t,
                    roi_size=tuple(int(x) for x in args.roi_size),
                    sw_batch_size=int(args.sw_batch),
                    predictor=model,
                )
        pred_raw = torch.argmax(split_vista_output_logits(logits)[0], dim=1)[0].detach().cpu().numpy().astype(np.int16)
        pred = maybe_apply_side_relabel_postprocess(pred_raw, args)
        gt, meta = priornet.load_volume(case.label)
        gt = np.asarray(gt, dtype=np.int16)
        spacing_mm = priornet.get_spacing_mm(meta)
        metrics = summarize_case_metrics(pred, gt, spacing_mm)
        sum_dice_cls1 += float(metrics["dice_cls1"])
        sum_dice_cls2 += float(metrics["dice_cls2"])
        sum_dice_fg += float(metrics["dice_fg"])
        sum_dice_mean += float(metrics["dice_mean"])
        sum_hd95_mean += float(metrics["hd95_mean"])

    num_cases = float(max(1, len(val_cases)))
    return {
        "epoch": int(epoch_idx),
        "num_cases": int(len(val_cases)),
        "dice_cls1": float(sum_dice_cls1 / num_cases),
        "dice_cls2": float(sum_dice_cls2 / num_cases),
        "dice_fg": float(sum_dice_fg / num_cases),
        "dice_mean": float(sum_dice_mean / num_cases),
        "hd95_mean": float(sum_hd95_mean / num_cases),
    }


def train_sup_stage(args) -> None:
    configure_torch_runtime(bool(args.deterministic))
    set_seed(int(args.seed))
    patch_monai_max_seed()
    if not args.init_checkpoint:
        raise ValueError("--init-checkpoint is required for --stage sup.")

    dataset_root = os.path.abspath(args.dataset_root)
    split_json = args.split_json or os.path.join(dataset_root, "splits.json")
    labeled_cases, _ = load_split_cases(dataset_root, split_json, args.split_ratio)
    val_cases = load_validation_cases(dataset_root, split_json, args.split_ratio, args.val_key)
    orientation_cases = list({case.case_id: case for case in [*labeled_cases, *val_cases]}.values())
    orientation_summary = summarize_orientation_consistency(orientation_cases, lateral_axis=int(args.lateral_axis))
    args.class1_is_high_x = bool(orientation_summary["increasing_toward"] == "left")

    prefix = build_output_prefix(args.split_ratio, args.run_name, "sup")
    run_dir = os.path.join(os.path.abspath(args.output_dir), prefix)
    os.makedirs(run_dir, exist_ok=True)
    best_ckpt = os.path.join(run_dir, "best_model.pth")
    best_metrics_json = os.path.join(run_dir, f"best_{args.val_key or 'validation'}_metrics.json")

    if torch.cuda.is_available():
        gpu_idx = parse_gpu_index(args.gpu)
        if gpu_idx >= torch.cuda.device_count():
            raise ValueError(f"--gpu={args.gpu} is out of range. Available GPU indices: 0..{torch.cuda.device_count() - 1}")
        torch.cuda.set_device(gpu_idx)
        device = torch.device(f"cuda:{gpu_idx}")
    else:
        device = torch.device("cpu")
    use_amp = bool(args.amp and device.type == "cuda")

    model = build_aorta_vista3d_model(in_channels=1, class_ids=(1, 2)).to(device)
    pretrained_ckpt = ensure_exists(os.path.abspath(args.init_checkpoint), "init checkpoint")
    pretrained_meta = load_pretrained_weights(model, pretrained_ckpt, device=device)
    print(
        f"Loaded checkpoint: {pretrained_ckpt} "
        f"(epoch={int(pretrained_meta.get('epoch', -1))}, split_ratio={pretrained_meta.get('args', {}).get('split_ratio', 'unknown')})"
    )
    optimizer = AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = make_cuda_grad_scaler(enabled=use_amp)
    sup_criterion = SideAwareDiceCELoss(
        lambda_dice=1.0,
        lambda_ce=1.0,
        lambda_swap=float(args.swap_loss_weight),
        lambda_centroid=float(args.centroid_loss_weight),
        centroid_margin=float(args.centroid_margin),
        lateral_axis=int(args.lateral_axis),
    )
    train_tf = build_train_transform(
        roi_size=args.roi_size,
        p_low=float(args.norm_p_low),
        p_high=float(args.norm_p_high),
        num_samples=int(args.num_samples),
    )
    labeled_data = [{"image": c.image, "label": c.label} for c in labeled_cases]
    labeled_loader = build_loader(
        data=labeled_data,
        transform=train_tf,
        batch_size=max(1, int(args.batchsize)),
        workers=max(0, int(args.workers)),
        drop_last=False,
    )
    if labeled_loader is None:
        raise RuntimeError("No labeled cases found for supervised training.")

    best_val_dice = float("nan")
    best_val_hd95 = float("inf")
    for epoch in range(1, int(args.max_epochs) + 1):
        current_lr = get_current_lr(epoch, args)
        for param_group in optimizer.param_groups:
            param_group["lr"] = float(current_lr)
        model.train()
        epoch_sup_loss = 0.0
        sup_steps = len(labeled_loader)
        optimizer.zero_grad(set_to_none=True)
        for batch_l in tqdm(labeled_loader, desc=f"Sup E{epoch}/{args.max_epochs}", leave=True, dynamic_ncols=True, mininterval=1.5):
            img_l = batch_l["image"].to(device=device, dtype=torch.float32)
            lbl_l = batch_l["label"].to(device=device, dtype=torch.long)
            amp_ctx = cuda_autocast(enabled=True) if use_amp else nullcontext()
            with amp_ctx:
                logit_l = model(img_l)
                loss_sup = sup_criterion(split_vista_output_logits(logit_l)[0], lbl_l)
            scaler.scale(loss_sup).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            epoch_sup_loss += float(loss_sup.item())
        epoch_sup_loss /= max(1, sup_steps)

        val_summary = None
        if len(val_cases) > 0 and (epoch % max(1, int(args.val_every)) == 0):
            val_summary = evaluate_supervised_validation(
                model=model,
                val_cases=val_cases,
                args=args,
                device=device,
                epoch_idx=epoch,
                use_amp=use_amp,
            )
            cur_dice = float(val_summary["dice_mean"])
            cur_hd95 = float(val_summary["hd95_mean"])
            if is_better_validation(cur_dice, cur_hd95, best_val_dice, best_val_hd95):
                best_val_dice = cur_dice
                best_val_hd95 = cur_hd95
                save_checkpoint(best_ckpt, model, optimizer, epoch, epoch_sup_loss, args, extra={"init_checkpoint": pretrained_ckpt, **val_summary})
                write_json(
                    best_metrics_json,
                    {
                        "checkpoint": os.path.basename(best_ckpt),
                        "epoch": int(val_summary["epoch"]),
                        "num_cases": int(val_summary["num_cases"]),
                        "dice_cls1": float(val_summary["dice_cls1"]),
                        "dice_cls2": float(val_summary["dice_cls2"]),
                        "dice_fg": float(val_summary["dice_fg"]),
                        "dice_mean": float(val_summary["dice_mean"]),
                        "hd95_mean": float(val_summary["hd95_mean"]),
                        "init_checkpoint": pretrained_ckpt,
                    },
                )

        if val_summary is not None:
            print(
                f"Epoch {epoch:03d} | sup={epoch_sup_loss:.4f} lr={current_lr:.2e} "
                f"val_dice_cls1={float(val_summary['dice_cls1']):.4f} "
                f"val_dice_cls2={float(val_summary['dice_cls2']):.4f} "
                f"val_dice_fg={float(val_summary['dice_fg']):.4f} "
                f"val_dice_mean={float(val_summary['dice_mean']):.4f}"
            )
        else:
            print(f"Epoch {epoch:03d} | sup={epoch_sup_loss:.4f} lr={current_lr:.2e}")


def evaluate_model_dice(
    model: torch.nn.Module,
    val_cases,
    args,
    device: torch.device,
    epoch_idx: int,
    use_amp: bool,
) -> Dict[str, object]:
    model.eval()
    if len(val_cases) == 0:
        return {
            "epoch": int(epoch_idx),
            "num_cases": 0,
            "dice_cls1": float("nan"),
            "dice_cls2": float("nan"),
            "dice_fg": float("nan"),
            "dice_mean": float("nan"),
        }

    sum_dice_cls1 = 0.0
    sum_dice_cls2 = 0.0
    sum_dice_fg = 0.0
    sum_dice_mean = 0.0
    for case in tqdm(val_cases, desc=f"Eval E{epoch_idx}", leave=True, dynamic_ncols=True, mininterval=1.5):
        vol, _ = priornet.load_volume(case.image)
        vol = np.asarray(vol, dtype=np.float32)
        image_norm = normalize_intensity_np(vol, args.norm_p_low, args.norm_p_high)
        image_t = torch.from_numpy(image_norm[None, None]).to(device)
        amp_ctx = cuda_autocast(enabled=True) if use_amp else nullcontext()
        with torch.no_grad():
            with amp_ctx:
                logits = sliding_window_inference(
                    image_t,
                    roi_size=tuple(int(x) for x in args.roi_size),
                    sw_batch_size=int(args.sw_batch),
                    predictor=model,
                )
        pred_raw = torch.argmax(split_vista_output_logits(logits)[0], dim=1)[0].detach().cpu().numpy().astype(np.int16)
        pred = maybe_apply_side_relabel_postprocess(pred_raw, args) if hasattr(args, "class1_is_high_x") else pred_raw
        gt, meta = priornet.load_volume(case.label)
        gt = np.asarray(gt, dtype=np.int16)
        metrics = summarize_case_metrics(pred, gt, priornet.get_spacing_mm(meta))
        sum_dice_cls1 += float(metrics["dice_cls1"])
        sum_dice_cls2 += float(metrics["dice_cls2"])
        sum_dice_fg += float(metrics["dice_fg"])
        sum_dice_mean += float(metrics["dice_mean"])
    num_cases = float(max(1, len(val_cases)))
    return {
        "epoch": int(epoch_idx),
        "num_cases": int(len(val_cases)),
        "dice_cls1": float(sum_dice_cls1 / num_cases),
        "dice_cls2": float(sum_dice_cls2 / num_cases),
        "dice_fg": float(sum_dice_fg / num_cases),
        "dice_mean": float(sum_dice_mean / num_cases),
    }


def apply_stage_defaults(args) -> None:
    if not getattr(args, "split_ratio", None):
        args.split_ratio = "10%" if str(args.stage) == "unsup" else "5%"
    if args.max_epochs is None:
        args.max_epochs = 150 if str(args.stage) == "unsup" else (100 if str(args.stage) == "sup" else 200)
    if args.lr is None:
        args.lr = 1e-4 if str(args.stage) == "unsup" else (1e-5 if str(args.stage) == "sup" else 3e-5)


def train(args) -> None:
    apply_stage_defaults(args)
    if str(args.stage) == "unsup":
        train_unsup_stage(args)
    elif str(args.stage) == "sup":
        train_sup_stage(args)
    elif str(args.stage) == "corefined":
        train_corefined_stage(args)
    else:
        raise ValueError(f"Unsupported stage: {args.stage}")


def parse_train_args():
    parser = argparse.ArgumentParser(description="APPR unified training entry")
    parser.add_argument("--stage", type=str, required=True, choices=["unsup", "sup", "corefined"])
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split-json", type=str, default=None, help="Default: <dataset-root>/splits.json")
    parser.add_argument("--split-ratio", type=str, default=None, help="5%%, 10%%, 20%%, 50%%, 100%%")
    parser.add_argument(
        "--val-key",
        type=str,
        default="test",
        help="Fixed validation list key in split json (e.g. test). Empty string disables validation.",
    )
    parser.add_argument("--val-every", type=int, default=1, help="Run validation every N epochs")

    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=1, help="RandCropByPosNegLabeld num_samples")
    parser.add_argument("--roi-size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sw-batch", type=int, default=1)

    parser.add_argument("--lr", type=float, default=None, help="Stage-specific default if omitted.")
    parser.add_argument("--lr-warmup-epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--unsup-weight", type=float, default=1.0)
    parser.add_argument("--corefined-weight", type=float, default=1.0)
    parser.add_argument("--swap-loss-weight", type=float, default=1.0)
    parser.add_argument("--centroid-loss-weight", type=float, default=0.25)
    parser.add_argument("--centroid-margin", type=float, default=0.05)
    parser.add_argument("--lateral-axis", type=int, default=-1)
    parser.add_argument(
        "--labeled-loss-weight",
        type=float,
        default=1.0,
        help="Relative loss weight applied to GT-guided labeled co-refined patches.",
    )
    parser.add_argument(
        "--unlabeled-loss-weight",
        type=float,
        default=0.25,
        help="Relative loss weight applied to pseudo-labeled unlabeled co-refined patches. Stable baseline default: 0.25.",
    )
    parser.add_argument(
        "--unlabeled-start-epoch",
        type=int,
        default=11,
        help="First epoch that allows unlabeled pseudo-guided co-refinement patches to enter training. Stable baseline default: 11.",
    )
    parser.add_argument(
        "--gt-only",
        action="store_true",
        help="Disable unlabeled co-refined training and keep only GT-guided labeled patches.",
    )
    parser.add_argument(
        "--unsup-only-epochs",
        type=int,
        default=0,
        help="Deprecated. Ignored because unlabeled co-refinement timing is controlled by --unlabeled-start-epoch.",
    )
    parser.add_argument("--unsup-accum-steps", type=int, default=0, help="0 means auto by labeled/unlabeled ratio")
    parser.add_argument("--max-unsup-accum-steps", type=int, default=32)
    parser.add_argument(
        "--disable-side-relabel-postprocess",
        action="store_true",
        help="Disable geometry-based cls1/cls2 relabeling on the predicted foreground during validation/inference.",
    )
    parser.add_argument(
        "--side-relabel-split",
        type=float,
        default=0.5,
        help="Normalized x split used by the geometry-based cls1/cls2 relabel postprocess.",
    )

    parser.add_argument("--norm-p-low", type=float, default=1.0)
    parser.add_argument("--norm-p-high", type=float, default=99.0)

    parser.add_argument("--pseudo-start-epoch", type=int, default=1)
    parser.add_argument("--pseudo-update-every", type=int, default=1)
    parser.add_argument("--pseudo-use-vista-guidance", action="store_true")
    parser.add_argument("--pseudo-vista-guidance-start-epoch", type=int, default=1)
    parser.add_argument("--pseudo-min-component", type=int, default=150)
    parser.add_argument(
        "--pseudo-train-min-vox",
        type=int,
        default=1000,
        help="Reject unlabeled pseudo patches whose foreground voxels are below this size. Stable baseline default: 1000.",
    )
    parser.add_argument(
        "--pseudo-train-max-vox",
        type=int,
        default=0,
        help="Reject unlabeled pseudo patches whose foreground voxels exceed this value. 0 disables the cap.",
    )
    parser.add_argument(
        "--pseudo-train-min-cls-vox",
        type=int,
        default=128,
        help="Reject unlabeled pseudo patches if a present cls1/cls2 region has fewer than this many voxels. Stable baseline default: 128.",
    )
    parser.add_argument(
        "--pseudo-require-both-classes",
        action="store_true",
        help="Require unlabeled pseudo patches to contain both cls1 and cls2 foreground before using them for training.",
    )
    parser.add_argument("--pseudo-fallback-min-vox", type=int, default=150)
    parser.add_argument("--pseudo-fg-anneal-epochs", type=int, default=30)
    parser.add_argument("--pseudo-fg-vista-reliance-start", type=float, default=0.0)
    parser.add_argument("--pseudo-fg-vista-reliance-end", type=float, default=1.0)
    parser.add_argument("--pseudo-fg-seed-thr-bonus", type=float, default=0.15)
    parser.add_argument("--pseudo-bg-seed-thr-bonus", type=float, default=0.05)
    parser.add_argument("--pseudo-structure-score-thr", type=float, default=0.30)
    parser.add_argument("--pseudo-side-relaxed-score-thr", type=float, default=0.18)
    parser.add_argument("--pseudo-side-assign-thr", type=float, default=0.55)
    parser.add_argument("--pseudo-class-anneal-epochs", type=int, default=30)
    parser.add_argument("--pseudo-vista-reliance-start", type=float, default=0.10)
    parser.add_argument("--pseudo-vista-reliance-end", type=float, default=0.95)
    parser.add_argument("--pseudo-midline-low", type=float, default=0.45)
    parser.add_argument("--pseudo-midline-high", type=float, default=0.55)
    parser.add_argument("--pseudo-component-class-margin", type=float, default=0.05)
    parser.add_argument(
        "--hold-vram-during-pseudo",
        action="store_true",
        default=True,
        help="Reserve supervised-phase GPU memory during pseudo-label generation to reduce VRAM drops.",
    )
    parser.add_argument(
        "--disable-hold-vram-during-pseudo",
        dest="hold_vram_during_pseudo",
        action="store_false",
        help="Disable pseudo-phase GPU memory reservation.",
    )
    parser.add_argument(
        "--pseudo-vram-headroom-mb",
        type=float,
        default=1024.0,
        help="Safety headroom kept free when reserving pseudo-phase VRAM.",
    )
    parser.add_argument(
        "--pseudo-vram-chunk-mb",
        type=int,
        default=256,
        help="Allocation chunk size for the pseudo-phase VRAM reservation buffer.",
    )

    parser.add_argument("--vista-fg-seed-thr", type=float, default=0.75)
    parser.add_argument("--vista-bg-seed-thr", type=float, default=0.90)
    parser.add_argument("--vista-cls-seed-thr", type=float, default=0.60)
    parser.add_argument("--vista-guidance-weight", type=float, default=0.75)
    parser.add_argument("--vista-seed-boost", type=float, default=2.0)

    parser.add_argument("--crf-iters", type=int, default=5)
    parser.add_argument("--crf-bilateral-weight", type=float, default=1.0)
    parser.add_argument("--crf-gaussian-weight", type=float, default=1.0)
    parser.add_argument("--crf-bilateral-spatial-sigma", type=float, default=5.0)
    parser.add_argument("--crf-bilateral-color-sigma", type=float, default=0.5)
    parser.add_argument("--crf-gaussian-spatial-sigma", type=float, default=5.0)
    parser.add_argument("--crf-update-factor", type=float, default=1.0)
    parser.add_argument("--crf-roi-prob-thr", type=float, default=0.10)
    parser.add_argument("--crf-roi-margin", type=int, default=6)

    parser.add_argument("--default-prob-thr", type=float, default=0.15)
    parser.add_argument("--prior-p-thr", type=float, default=0.10)
    parser.add_argument("--use-prior-strong-p", dest="use_prior_strong_p", action="store_true", default=True)
    parser.add_argument("--disable-prior-strong-p", dest="use_prior_strong_p", action="store_false")
    parser.add_argument("--sweep-start", type=float, default=0.05)
    parser.add_argument("--sweep-end", type=float, default=0.55)
    parser.add_argument("--sweep-step", type=float, default=0.02)
    parser.add_argument("--guidance-min-vox", type=int, default=128)

    parser.add_argument("--exclude-midline", dest="exclude_midline", action="store_true", default=True)
    parser.add_argument("--keep-midline", dest="exclude_midline", action="store_false")

    parser.add_argument("--prior-component-prob-thr", type=float, default=0.20)
    parser.add_argument("--prior-prob-gamma", type=float, default=1.0)
    parser.add_argument("--nonair-weight", type=float, default=0.25)
    parser.add_argument("--nonair-seed-weight", type=float, default=0.20)
    parser.add_argument("--nonair-center-q", type=float, default=0.55)
    parser.add_argument("--nonair-sigma-hu", type=float, default=120.0)
    parser.add_argument("--nonair-bone-q", type=float, default=0.92)
    parser.add_argument("--nonair-gate-floor", type=float, default=0.0)

    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--init-checkpoint", type=str, default=None, help="Required for stage sup/corefined.")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--gpu", type=str, default="3", help="GPU index, e.g. 0 or cuda:0")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic PyTorch/cuDNN runtime settings for reproducibility control experiments.",
    )
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_train_args()
    train(args)
