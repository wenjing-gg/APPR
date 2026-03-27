from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss


def _ensure_label_shape(labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    if labels.ndim == logits.ndim - 1:
        labels = labels.unsqueeze(1)
    if labels.ndim != logits.ndim:
        raise ValueError(f"Expected labels with ndim={logits.ndim} or {logits.ndim - 1}, got {labels.ndim}.")
    return labels.long()


def _masked_mean(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    valid = valid_mask.to(dtype=values.dtype)
    denom = valid.sum().clamp_min(1.0)
    return (values * valid).sum() / denom


class SideAwareDiceCELoss(nn.Module):
    def __init__(
        self,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
        lambda_swap: float = 1.0,
        lambda_centroid: float = 0.25,
        centroid_margin: float = 0.05,
        lateral_axis: int = -1,
    ) -> None:
        super().__init__()
        self.base_loss = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            lambda_dice=float(lambda_dice),
            lambda_ce=float(lambda_ce),
        )
        self.lambda_swap = float(lambda_swap)
        self.lambda_centroid = float(lambda_centroid)
        self.centroid_margin = float(centroid_margin)
        self.lateral_axis = int(lateral_axis)
        self.eps = 1e-6

    def _get_spatial_axis(self, target: torch.Tensor) -> int:
        axis = self.lateral_axis
        if axis < 0:
            axis += target.ndim
        if axis <= 0 or axis >= target.ndim:
            raise ValueError(
                f"lateral_axis={self.lateral_axis} resolves to axis={axis}, but target shape is {tuple(target.shape)}."
            )
        return axis

    def _cross_swap_loss(
        self,
        cls1_prob: torch.Tensor,
        cls2_prob: torch.Tensor,
        gt_cls1: torch.Tensor,
        gt_cls2: torch.Tensor,
    ) -> torch.Tensor:
        spatial_dims: Tuple[int, ...] = tuple(range(1, gt_cls1.ndim))
        cls1_mass = gt_cls1.sum(dim=spatial_dims)
        cls2_mass = gt_cls2.sum(dim=spatial_dims)

        cls1_on_cls2 = (cls1_prob * gt_cls2).sum(dim=spatial_dims) / cls2_mass.clamp_min(1.0)
        cls2_on_cls1 = (cls2_prob * gt_cls1).sum(dim=spatial_dims) / cls1_mass.clamp_min(1.0)

        loss_12 = _masked_mean(cls1_on_cls2, cls2_mass > 0)
        loss_21 = _masked_mean(cls2_on_cls1, cls1_mass > 0)
        return 0.5 * (loss_12 + loss_21)

    def _centroid_loss(
        self,
        cls1_prob: torch.Tensor,
        cls2_prob: torch.Tensor,
        gt_cls1: torch.Tensor,
        gt_cls2: torch.Tensor,
    ) -> torch.Tensor:
        spatial_dims: Tuple[int, ...] = tuple(range(1, gt_cls1.ndim))
        axis = self._get_spatial_axis(gt_cls1)

        coord = torch.linspace(0.0, 1.0, steps=gt_cls1.shape[axis], device=gt_cls1.device, dtype=gt_cls1.dtype)
        coord_shape = [1] * gt_cls1.ndim
        coord_shape[axis] = gt_cls1.shape[axis]
        coord = coord.view(coord_shape)

        pred_mass_1 = cls1_prob.sum(dim=spatial_dims).clamp_min(self.eps)
        pred_mass_2 = cls2_prob.sum(dim=spatial_dims).clamp_min(self.eps)
        gt_mass_1 = gt_cls1.sum(dim=spatial_dims)
        gt_mass_2 = gt_cls2.sum(dim=spatial_dims)

        pred_centroid_1 = (cls1_prob * coord).sum(dim=spatial_dims) / pred_mass_1
        pred_centroid_2 = (cls2_prob * coord).sum(dim=spatial_dims) / pred_mass_2
        gt_centroid_1 = (gt_cls1 * coord).sum(dim=spatial_dims) / gt_mass_1.clamp_min(self.eps)
        gt_centroid_2 = (gt_cls2 * coord).sum(dim=spatial_dims) / gt_mass_2.clamp_min(self.eps)

        align = (pred_centroid_1 - gt_centroid_1).abs() + (pred_centroid_2 - gt_centroid_2).abs()

        gt_direction = torch.sign(gt_centroid_1 - gt_centroid_2)
        gt_direction = torch.where(gt_direction == 0, torch.ones_like(gt_direction), gt_direction)
        pred_signed_gap = (pred_centroid_1 - pred_centroid_2) * gt_direction
        order = F.relu(self.centroid_margin - pred_signed_gap)

        valid = (gt_mass_1 > 0) & (gt_mass_2 > 0)
        return _masked_mean(align + order, valid)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = _ensure_label_shape(labels, logits)
        multiclass_logits = logits[:, :3]

        total_loss = self.base_loss(multiclass_logits, labels)
        if self.lambda_swap <= 0.0 and self.lambda_centroid <= 0.0:
            return total_loss

        probs = torch.softmax(multiclass_logits, dim=1)
        target = labels[:, 0]
        gt_cls1 = (target == 1).to(dtype=probs.dtype)
        gt_cls2 = (target == 2).to(dtype=probs.dtype)

        if self.lambda_swap > 0.0:
            total_loss = total_loss + self.lambda_swap * self._cross_swap_loss(
                cls1_prob=probs[:, 1],
                cls2_prob=probs[:, 2],
                gt_cls1=gt_cls1,
                gt_cls2=gt_cls2,
            )

        if self.lambda_centroid > 0.0:
            total_loss = total_loss + self.lambda_centroid * self._centroid_loss(
                cls1_prob=probs[:, 1],
                cls2_prob=probs[:, 2],
                gt_cls1=gt_cls1,
                gt_cls2=gt_cls2,
            )

        return total_loss
