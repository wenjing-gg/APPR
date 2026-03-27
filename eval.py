import argparse
import json
import os
from types import SimpleNamespace
from typing import Dict, Optional

import torch

from VISTA3D import build_aorta_vista3d_model
from dataset import DEFAULT_DATASET_ROOT, ensure_exists, load_validation_cases, summarize_orientation_consistency
from train import evaluate_model_dice, parse_gpu_index


def load_checkpoint(path: str) -> Dict[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def choose_device(gpu_arg: Optional[str]) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    gpu_value = "0" if gpu_arg is None else str(gpu_arg)
    gpu_idx = parse_gpu_index(gpu_value)
    gpu_count = torch.cuda.device_count()
    if gpu_idx >= gpu_count:
        raise ValueError(f"--gpu={gpu_value} is out of range. Available GPU indices: 0..{gpu_count - 1}")
    torch.cuda.set_device(gpu_idx)
    return torch.device(f"cuda:{gpu_idx}")


def build_eval_args(cli_args, ckpt_args: Dict[str, object], orientation_summary: Dict[str, object]) -> SimpleNamespace:
    payload = dict(ckpt_args) if isinstance(ckpt_args, dict) else {}
    payload["norm_p_low"] = float(cli_args.norm_p_low if cli_args.norm_p_low is not None else payload.get("norm_p_low", 1.0))
    payload["norm_p_high"] = float(cli_args.norm_p_high if cli_args.norm_p_high is not None else payload.get("norm_p_high", 99.0))
    roi_size = cli_args.roi_size if cli_args.roi_size is not None else payload.get("roi_size", [96, 96, 96])
    payload["roi_size"] = [int(x) for x in roi_size]
    payload["sw_batch"] = int(cli_args.sw_batch if cli_args.sw_batch is not None else payload.get("sw_batch", 1))
    payload["class1_is_high_x"] = bool(orientation_summary["increasing_toward"] == "left")
    payload["disable_side_relabel_postprocess"] = bool(getattr(cli_args, "disable_side_relabel_postprocess", False))
    payload["side_relabel_split"] = float(payload.get("side_relabel_split", 0.5))
    return SimpleNamespace(**payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal final evaluation for APPR checkpoints")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split-json", type=str, default=None, help="Default: <dataset-root>/splits.json")
    parser.add_argument("--split-ratio", type=str, required=True)
    parser.add_argument("--eval-key", type=str, default="test")
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--roi-size", type=int, nargs=3, default=None)
    parser.add_argument("--sw-batch", type=int, default=None)
    parser.add_argument("--norm-p-low", type=float, default=None)
    parser.add_argument("--norm-p-high", type=float, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--disable-side-relabel-postprocess", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = ensure_exists(os.path.abspath(args.checkpoint), "checkpoint")
    dataset_root = os.path.abspath(args.dataset_root)
    split_json = args.split_json or os.path.join(dataset_root, "splits.json")
    split_json = ensure_exists(split_json, "split json")
    device = choose_device(args.gpu)
    use_amp = bool(args.amp and device.type == "cuda")

    ckpt = load_checkpoint(checkpoint_path)
    ckpt_args = ckpt.get("args", {})
    eval_cases = load_validation_cases(dataset_root, split_json, args.split_ratio, args.eval_key)
    if len(eval_cases) == 0:
        raise RuntimeError(f"No cases found for eval_key={args.eval_key!r}")
    orientation_summary = summarize_orientation_consistency(eval_cases, lateral_axis=-1)
    eval_args = build_eval_args(args, ckpt_args if isinstance(ckpt_args, dict) else {}, orientation_summary)

    model = build_aorta_vista3d_model(in_channels=1, class_ids=(1, 2))
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)

    summary = evaluate_model_dice(
        model=model,
        val_cases=eval_cases,
        args=eval_args,
        device=device,
        epoch_idx=int(ckpt.get("epoch", 0)),
        use_amp=use_amp,
    )

    payload = {
        "checkpoint": checkpoint_path,
        "split_ratio": str(args.split_ratio),
        "eval_split": str(args.eval_key),
        "device": str(device),
        "epoch": int(summary["epoch"]),
        "num_cases": int(summary["num_cases"]),
        "dice_cls1": float(summary["dice_cls1"]),
        "dice_cls2": float(summary["dice_cls2"]),
        "dice_fg": float(summary["dice_fg"]),
        "dice_mean": float(summary["dice_mean"]),
    }
    if args.output_json:
        out_json = os.path.abspath(args.output_json)
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
