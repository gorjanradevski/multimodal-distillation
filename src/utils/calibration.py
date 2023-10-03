import torch
import torch.nn.functional as F
from torch import nn
from yacs.config import CfgNode


class _ECELoss(nn.Module):
    """
    https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py#L78
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


class CalibrationEvaluator:
    def __init__(self, cfg: CfgNode):
        self.cfg = cfg
        # Get action names to have only one evaluator
        self.action_names = [
            action
            for action in self.cfg.ACTION_WEIGHTS.keys()
            if self.cfg.ACTION_WEIGHTS[action]
        ]
        # Aggregators
        self.logits = {}
        self.labels = {}
        for action_name in self.action_names:
            self.logits[action_name] = []
            self.labels[action_name] = []
        # Evaluation criterion
        self.ece_criterion = _ECELoss()

    def process(self, model_output, labels):
        for action_name in self.action_names:
            self.logits[action_name].append(model_output[action_name])
            self.labels[action_name].append(labels[action_name])

    def evaluate(self):
        calibration_metrics = {}
        for action_name in self.action_names:
            logits = torch.cat(self.logits[action_name], dim=0)
            # Because of the number of temporal clips x spatial crops
            logits = logits.mean(1)
            labels = torch.cat(self.labels[action_name], dim=0)
            calibration_metrics[action_name] = (
                self.ece_criterion(logits=logits, labels=labels).item() * 100.0
            )

        return calibration_metrics
