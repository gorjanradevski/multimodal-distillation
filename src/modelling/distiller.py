from os.path import join as pjoin
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn
from yacs.config import CfgNode

from modelling.models import model_factory
from utils.setup import get_cfg_defaults


class DistillationCriterion:
    def __init__(self, cfg):
        self.cfg = cfg
        self.gt_criterion = nn.CrossEntropyLoss()
        self.action_names = [
            action
            for action in self.cfg.ACTION_WEIGHTS.keys()
            if self.cfg.ACTION_WEIGHTS[action]
        ]

    def measure_temp_scaled_kl_loss(self, student_logits, teacher_logits):
        loss = 0
        for action_name in self.action_names:
            target = F.log_softmax(
                teacher_logits[action_name] / self.cfg.TEMPERATURE, dim=-1
            )
            pred = F.log_softmax(
                student_logits[action_name] / self.cfg.TEMPERATURE, dim=-1
            )
            cur_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)(
                pred, target
            )
            cur_loss = cur_loss * (self.cfg.TEMPERATURE ** 2)
            loss += cur_loss
        # Average loss
        loss = loss / len(self.action_names)

        return loss

    def measure_gt_loss(self, student_logits, gt_labels):
        total_loss = 0
        # Aggregate losses
        for action_name in self.action_names:
            loss = (
                self.gt_criterion(student_logits[action_name], gt_labels[action_name])
                * self.cfg.ACTION_WEIGHTS[action_name]
            )
            total_loss += loss

        return total_loss / len(self.action_names)

    def __call__(self, student_logits, teacher_logits, batch):
        # Measure losses
        distillation_loss = self.measure_temp_scaled_kl_loss(
            student_logits, teacher_logits
        )
        gt_loss = self.measure_gt_loss(student_logits, batch["labels"])
        # Compute total loss
        total_loss = (
            distillation_loss * self.cfg.LOSS_WEIGHTS.DISTILLATION
            + gt_loss * self.cfg.LOSS_WEIGHTS.GROUND_TRUTH
        )

        return total_loss


class TeacherEnsemble(nn.Module):
    def __init__(self, cfg: CfgNode):
        super(TeacherEnsemble, self).__init__()
        # Prepare teachers
        self.cfg = cfg
        teachers = {}
        for teacher_name in self.cfg.TEACHERS.keys():
            # Skip empty teachers
            if not self.cfg.TEACHERS.get(teacher_name):
                continue
            teacher_cfg = get_cfg_defaults()
            teacher_cfg.merge_from_file(
                pjoin(self.cfg.TEACHERS.get(teacher_name), "config.yaml")
            )
            teacher = model_factory[teacher_cfg.MODEL_NAME](teacher_cfg)
            checkpoint = torch.load(
                pjoin(self.cfg.TEACHERS.get(teacher_name), "model_checkpoint.pt"),
                map_location="cpu",
            )
            from collections import OrderedDict

            unwrapped_checkpoint = OrderedDict()
            prefix = "_orig_mod."
            for key in checkpoint.keys():
                if key.startswith(prefix):
                    unwrapped_checkpoint[key[len(prefix) :]] = checkpoint[key]
                else:
                    unwrapped_checkpoint[key] = checkpoint[key]
            teacher.load_state_dict(unwrapped_checkpoint)
            teacher.train(False)
            teachers[teacher_name] = teacher
        self.teachers = nn.ModuleDict(teachers)
        # Establish initial weights
        assert self.cfg.DISTILLATION_WEIGHTING_SCHEME in [
            None,
            "per-sample",
            "full-dataset",
        ]
        weights = {}
        for action_name, action_num in self.cfg.TOTAL_ACTIONS.items():
            if action_num is not None:
                weights[action_name] = nn.Parameter(
                    torch.full(
                        size=(len(teachers),), fill_value=1 / len(self.teachers)
                    ),
                    requires_grad=False,
                )
        self.weights = nn.ParameterDict(weights)

    def get_losses_wrt_labels(self, logits: torch.Tensor, labels: torch.Tensor):
        b, n_t, c = logits.size()
        # Prepare logits & labels
        logits = logits.reshape(-1, c)
        # BE CAREFULL, repeat would be wrong here!
        labels = labels.repeat_interleave(n_t)
        # Compute weights: [Batch_Size, Num_Teachers]
        per_teacher_losses = F.cross_entropy(logits, labels, reduction="none")
        # Reshape back in original shape
        per_teacher_losses = per_teacher_losses.reshape(b, n_t)

        return per_teacher_losses

    def get_per_sample_weights(self, logits: torch.Tensor, labels: torch.Tensor):
        per_teacher_losses = self.get_losses_wrt_labels(logits, labels)
        return F.softmin(per_teacher_losses / self.cfg.WEIGHTS_TEMPERATURE, dim=-1)

    def get_teacher_logits(self, batch):
        teacher_outputs: Dict[str, Dict[str, torch.Tensor]] = {}
        for teacher_name in self.cfg.TEACHERS.keys():
            # Skip empty teachers
            if not self.cfg.TEACHERS.get(teacher_name):
                continue
            teacher_outputs[teacher_name] = self.teachers[teacher_name](batch)
        # Gather teacher logits
        teacher_logits: Dict[str, List[torch.Tensor]] = {}
        for teacher_output in teacher_outputs.values():
            for action_name in teacher_output.keys():
                if action_name not in teacher_logits:
                    teacher_logits[action_name] = []
                teacher_logits[action_name].append(teacher_output[action_name])

        return teacher_logits

    def get_weights(
        self, action_name: str, logits: torch.Tensor, labels: Dict[str, torch.Tensor]
    ):
        # Weights obtained dynamically per-sample
        if self.cfg.DISTILLATION_WEIGHTING_SCHEME == "per-sample":
            weights = self.get_per_sample_weights(logits, labels[action_name])
        # Weights were updated during set_teacher_weights in patient distill
        elif self.cfg.DISTILLATION_WEIGHTING_SCHEME == "full-dataset":
            weights = self.weights[action_name]
        # The initial weights are set-up as 1 / num_teachers
        elif self.cfg.DISTILLATION_WEIGHTING_SCHEME is None:
            weights = self.weights[action_name]

        return weights

    @torch.no_grad()
    def forward(self, batch):
        teacher_logits = self.get_teacher_logits(batch)
        # (Weighed) average of teacher logits
        for action_name in teacher_logits:
            # [Batch_Size, Num_Teachers, Num_Classes]
            logits = torch.cat(
                [t_o.unsqueeze(1) for t_o in teacher_logits[action_name]], dim=1
            )
            weights = self.get_weights(action_name, logits, batch["labels"])
            # [Batch_Size, Num_Teachers]
            logits *= weights.unsqueeze(-1)
            # Average teacher predictions
            teacher_logits[action_name] = logits.sum(1)

        return teacher_logits
