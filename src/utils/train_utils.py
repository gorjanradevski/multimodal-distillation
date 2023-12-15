import json

import torch
from torch import optim


def get_linear_schedule_with_warmup(
    optimizer: optim.Optimizer, num_warmup_steps: int, num_training_steps: int
):
    # https://huggingface.co/transformers/_modules/transformers/optimization.html#get_linear_schedule_with_warmup
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def move_batch_to_device(batch, device):
    tmp_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            tmp_batch[k] = v.to(device)
        elif isinstance(v, dict):
            tmp_batch[k] = {}
            for inside_k, inside_v in v.items():
                tmp_batch[k][inside_k] = inside_v.to(device)
        else:
            tmp_batch[k] = v

    return tmp_batch


class EpochHandler:
    def __init__(self, epoch: int = 0):
        self.epoch = epoch

    def save_state(self, path: str):
        json.dump({"epoch": self.epoch}, open(path, "w"))

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def load_state(self, path: str):
        checkpoint = json.load(open(path))
        self.epoch = checkpoint["epoch"]
