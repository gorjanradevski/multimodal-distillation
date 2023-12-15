from torch import nn
from yacs.config import CfgNode


class LossesModule:
    def __init__(self, cfg: CfgNode):
        self.cfg = cfg
        self.criterions = {
            "charades-ego": nn.BCEWithLogitsLoss(),
            "something-something": nn.CrossEntropyLoss(),
            "egtea-gaze": nn.CrossEntropyLoss(),
            "EPIC-KITCHENS": nn.CrossEntropyLoss(),
        }
        self.action_names = [
            action
            for action in self.cfg.ACTION_WEIGHTS.keys()
            if self.cfg.ACTION_WEIGHTS[action]
        ]

    def __call__(self, model_output, batch):
        total_loss = 0
        # Aggregate losses
        for action_name in self.action_names:
            loss = (
                self.criterions[self.cfg.TRAIN_DATASET_NAME](
                    model_output[action_name],
                    batch["labels"][action_name],
                )
                * self.cfg.ACTION_WEIGHTS[action_name]
            )
            total_loss += loss

        return total_loss / len(self.action_names)
