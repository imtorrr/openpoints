# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss

Modified by Guocheng Qian
"""

import torch
from torch.nn import functional as F
from .build import LOSS, build_criterion_from_cfg


@LOSS.register_module()
class DistillLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """

    def __init__(
        self,
        base_criterion_args,
        distill_type: str = "hard",
        alpha: float = 0.5,
        tau: float = 10.0,
        **kwargs,
    ):
        super().__init__()
        self.base_criterion = build_criterion_from_cfg(base_criterion_args)
        assert distill_type in ["none", "soft", "hard"]
        self.distill_type = distill_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels, teacher):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distill_type == "none":
            return base_loss

        if outputs_kd is None:
            raise ValueError(
                "When knowledge distillation is enabled, the model is "
                "expected to return a Tuple[Tensor, Tensor] with the output of the "
                "class_token and the dist_token"
            )
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher.eval()
            teacher_outputs = teacher(inputs)

        if self.distill_type == "soft":
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = (
                F.kl_div(
                    F.log_softmax(outputs_kd / T, dim=1),
                    # We provide the teacher's targets in log probability because we use log_target=True
                    # (as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
                    # but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
                    F.log_softmax(teacher_outputs / T, dim=1),
                    reduction="sum",
                    log_target=True,
                )
                * (T * T)
                / outputs_kd.numel()
            )
            # We divide by outputs_kd.numel() to have the legacy PyTorch behavior.
            # But we also experiments output_kd.size(0)
            # see issue 61(https://github.com/facebookresearch/deit/issues/61) for more details
        elif self.distill_type == "hard":
            distillation_loss = F.cross_entropy(
                outputs_kd, teacher_outputs.argmax(dim=1)
            )

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss
