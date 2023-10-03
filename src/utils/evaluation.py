import csv
from typing import Dict

import numpy as np
import torch
from yacs.config import CfgNode


class EvaluatorAccuracy:
    def __init__(self, total_instances: int):
        self.total_instances = total_instances
        # FIXME: Hacky solution because of multi-GPU evaluation
        self.data = {
            "top1_corr": np.zeros((int(self.total_instances * 1.1))),
            "top5_corr": np.zeros((int(self.total_instances * 1.1))),
        }
        self.best_acc = 0.0
        self.processed_instances = 0

    def reset(self):
        self.data = {}
        # FIXME: Hacky solution because of multi-GPU evaluation
        self.data = {
            "top1_corr": np.zeros((int(self.total_instances * 1.1))),
            "top5_corr": np.zeros((int(self.total_instances * 1.1))),
        }
        self.processed_instances = 0

    def process(self, logits: torch.Tensor, labels: torch.Tensor):
        assert (
            len(logits.shape) == 3
        ), "The shape of logits must be in format [bs, num_test_clips * num_test_crops, total_classes]"
        num_instances = logits.shape[0]
        logits = logits.mean(1)
        self.data["top1_corr"][
            self.processed_instances : self.processed_instances + num_instances
        ] = (logits.argmax(-1) == labels).int()
        self.data["top5_corr"][
            self.processed_instances : self.processed_instances + num_instances
        ] = ((logits.topk(k=5).indices == labels.unsqueeze(1)).any(dim=1)).int()
        self.processed_instances += num_instances

    def evaluate(self):
        top1_acc = (
            self.data["top1_corr"].sum() / self.processed_instances
            if self.processed_instances
            else 0.0
        )
        top5_acc = (
            self.data["top5_corr"].sum() / self.processed_instances
            if self.processed_instances
            else 0.0
        )
        metrics = {
            "top1_acc": round(top1_acc * 100, 2),
            "top5_acc": round(top5_acc * 100, 2),
        }

        return metrics

    def evaluate_verbose(self):
        metrics = {}
        # Get metrics
        for m_name, m_value in self.evaluate().items():
            conf = (
                (m_value * (100 - m_value) / self.processed_instances) ** 0.5
                if self.processed_instances
                else 0.0
            )
            conf = round(conf, 2)
            metrics[m_name] = f"{m_value} +/- {conf}"

        return metrics

    def is_best(self):
        metrics = self.evaluate()
        # Get currect accuracy
        cur_accuracy = sum(
            [metrics[accuracy_type] for accuracy_type in metrics.keys()]
        ) / len(metrics)
        # Validate whether it's the best model
        if cur_accuracy > self.best_acc:
            self.best_acc = cur_accuracy
            return True
        return False


class EvaluatorAND:
    def __init__(self, evaluator1: EvaluatorAccuracy, evaluator2: EvaluatorAccuracy):
        self.evaluator1 = evaluator1
        self.evaluator2 = evaluator2
        assert (
            self.evaluator1.total_instances == self.evaluator2.total_instances
        ), "Two evaluators need to have the same number of total instances."

    def evaluate(self):
        assert (
            self.evaluator1.processed_instances == self.evaluator2.processed_instances
        ), "Two evaluators need to have processed the same number of instances."
        processed_instances = self.evaluator1.processed_instances
        corrects1 = self.evaluator1.data["top1_corr"]
        corrects2 = self.evaluator2.data["top1_corr"]

        corrects = ((corrects1 + corrects2) / 2 == 1).astype(int)
        accuracy = corrects.sum() / processed_instances
        metrics = {"top1_acc": round(accuracy * 100, 2)}

        return metrics

    def evaluate_verbose(self):
        assert (
            self.evaluator1.processed_instances == self.evaluator2.processed_instances
        ), "Two evaluators need to have processed the same number of instances."
        processed_instances = self.evaluator1.processed_instances
        m_value = self.evaluate()["top1_acc"]
        conf = (
            (m_value * (100 - m_value) / processed_instances) ** 0.5
            if processed_instances
            else 0.0
        )
        conf = round(conf, 2)
        metrics = {"top1_acc": f"{m_value} +/- {conf}"}

        return metrics


class EvaluatorSomething(EvaluatorAccuracy):
    def __init__(self, total_instances, cfg: CfgNode):
        self.cfg = cfg
        super().__init__(
            total_instances,
        )

    def process(
        self, model_output: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]
    ):
        # Prepare logits & labels
        logits = model_output["ACTION"]
        labels = labels["ACTION"]
        super().process(logits, labels)


class EvaluatorEpic:
    def __init__(self, total_instances: int, cfg: CfgNode):
        self.cfg = cfg
        self.noun_evaluator = EvaluatorAccuracy(
            total_instances,
        )
        self.verb_evaluator = EvaluatorAccuracy(
            total_instances,
        )
        self.action_evaluator = EvaluatorAND(self.noun_evaluator, self.verb_evaluator)
        self.best_acc = 0.0
        # Prepare EPIC tail nouns, if provided
        if self.cfg.EPIC_TAIL_NOUNS_PATH is not None:
            assert (
                cfg.DATASET_VERSION == 100
            ), "Tail classes only available on 'EPIC-KITCHENS 100'"
            with open(cfg.EPIC_TAIL_NOUNS_PATH, newline="") as csvfile:
                reader = csv.reader(csvfile, delimiter=" ", quotechar="|")
                tail_nouns = list(reader)[1:]
            self.tail_nouns = torch.tensor(
                [int(item[0]) for item in tail_nouns]
            ).unsqueeze(0)
            self.tail_noun_evaluator = EvaluatorAccuracy(
                total_instances,
            )
        # Prepare EPIC tail verbs, if provided
        if self.cfg.EPIC_TAIL_VERBS_PATH is not None:
            assert (
                cfg.DATASET_VERSION == 100
            ), "Tail classes only available on 'EPIC-KITCHENS 100'"
            with open(cfg.EPIC_TAIL_VERBS_PATH, newline="") as csvfile:
                reader = csv.reader(csvfile, delimiter=" ", quotechar="|")
                tail_verbs = list(reader)[1:]
            self.tail_verbs = torch.tensor(
                [int(item[0]) for item in tail_verbs]
            ).unsqueeze(0)
            self.tail_verb_evaluator = EvaluatorAccuracy(
                total_instances,
            )

        if hasattr(self, "tail_noun_evaluator") and hasattr(
            self, "tail_verb_evaluator"
        ):
            self.tail_noun_evaluator_aux_AND = EvaluatorAccuracy(
                total_instances,
            )
            self.tail_verb_evaluator_aux_AND = EvaluatorAccuracy(
                total_instances,
            )
            self.tail_action_evaluator_AND = EvaluatorAND(
                self.tail_noun_evaluator_aux_AND, self.tail_verb_evaluator_aux_AND
            )
            self.tail_noun_evaluator_aux_OR = EvaluatorAccuracy(
                total_instances,
            )
            self.tail_verb_evaluator_aux_OR = EvaluatorAccuracy(
                total_instances,
            )
            self.tail_action_evaluator_OR = EvaluatorAND(
                self.tail_noun_evaluator_aux_OR, self.tail_verb_evaluator_aux_OR
            )

    def reset(self):
        self.noun_evaluator.reset()
        self.verb_evaluator.reset()
        if hasattr(self, "tail_noun_evaluator"):
            self.tail_noun_evaluator.reset()
        if hasattr(self, "tail_verb_evaluator"):
            self.tail_verb_evaluator.reset()
        if hasattr(self, "tail_action_evaluator_AND"):
            self.tail_noun_evaluator_aux_AND.reset()
            self.tail_verb_evaluator_aux_AND.reset()
        if hasattr(self, "tail_action_evaluator_OR"):
            self.tail_noun_evaluator_aux_OR.reset()
            self.tail_verb_evaluator_aux_OR.reset()

    def process(
        self, model_output: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]
    ):
        self.noun_evaluator.process(model_output["NOUN"], labels["NOUN"])
        self.verb_evaluator.process(model_output["VERB"], labels["VERB"])
        if hasattr(self, "tail_noun_evaluator"):
            labels_noun = labels["NOUN"].unsqueeze(1)
            active_batch_indices_noun = (labels_noun == self.tail_nouns).any(dim=1)
            self.tail_noun_evaluator.process(
                model_output["NOUN"][active_batch_indices_noun],
                labels["NOUN"][active_batch_indices_noun],
            )
        if hasattr(self, "tail_verb_evaluator"):
            labels_verb = labels["VERB"].unsqueeze(1)
            active_batch_indices_verb = (labels_verb == self.tail_verbs).any(dim=1)
            self.tail_verb_evaluator.process(
                model_output["VERB"][active_batch_indices_verb],
                labels["VERB"][active_batch_indices_verb],
            )
        if hasattr(self, "tail_action_evaluator_AND"):
            actve_batch_indices_action_AND = torch.logical_and(
                active_batch_indices_noun, active_batch_indices_verb
            )
            self.tail_noun_evaluator_aux_AND.process(
                model_output["NOUN"][actve_batch_indices_action_AND],
                labels["NOUN"][actve_batch_indices_action_AND],
            )
            self.tail_verb_evaluator_aux_AND.process(
                model_output["VERB"][actve_batch_indices_action_AND],
                labels["VERB"][actve_batch_indices_action_AND],
            )

        if hasattr(self, "tail_action_evaluator_OR"):
            actve_batch_indices_action_OR = torch.logical_or(
                active_batch_indices_noun, active_batch_indices_verb
            )
            self.tail_noun_evaluator_aux_OR.process(
                model_output["NOUN"][actve_batch_indices_action_OR],
                labels["NOUN"][actve_batch_indices_action_OR],
            )
            self.tail_verb_evaluator_aux_OR.process(
                model_output["VERB"][actve_batch_indices_action_OR],
                labels["VERB"][actve_batch_indices_action_OR],
            )

    def evaluate(self):
        noun_metrics = self.noun_evaluator.evaluate()
        verb_metrics = self.verb_evaluator.evaluate()
        action_metrics = self.action_evaluator.evaluate()
        if hasattr(self, "tail_noun_evaluator"):
            tail_noun_metrics = self.tail_noun_evaluator.evaluate()
        else:
            tail_noun_metrics = {"top1_acc": "N/A", "top5_acc": "N/A"}
        if hasattr(self, "tail_verb_evaluator"):
            tail_verb_metrics = self.tail_verb_evaluator.evaluate()
        else:
            tail_verb_metrics = {"top1_acc": "N/A", "top5_acc": "N/A"}
        if hasattr(self, "tail_action_evaluator_AND"):
            tail_action_metrics_AND = self.tail_action_evaluator_AND.evaluate()
        else:
            tail_action_metrics_AND = {"top1_acc": "N/A"}
        if hasattr(self, "tail_action_evaluator_OR"):
            tail_action_metrics_OR = self.tail_action_evaluator_OR.evaluate()
        else:
            tail_action_metrics_OR = {"top1_acc": "N/A"}

        return {
            "noun_acc": noun_metrics["top1_acc"],
            "verb_acc": verb_metrics["top1_acc"],
            "action_acc": action_metrics["top1_acc"],
            "tail_noun_acc": tail_noun_metrics["top1_acc"],
            "tail_verb_acc": tail_verb_metrics["top1_acc"],
            "tail_action_AND_acc": tail_action_metrics_AND["top1_acc"],
            "tail_action_OR_acc": tail_action_metrics_OR["top1_acc"],
        }

    def evaluate_verbose(self):
        noun_metrics = self.noun_evaluator.evaluate_verbose()
        verb_metrics = self.verb_evaluator.evaluate_verbose()
        action_metrics = self.action_evaluator.evaluate_verbose()
        if hasattr(self, "tail_noun_evaluator"):
            tail_noun_metrics = self.tail_noun_evaluator.evaluate_verbose()
        else:
            tail_noun_metrics = {"top1_acc": "N/A", "top5_acc": "N/A"}
        if hasattr(self, "tail_verb_evaluator"):
            tail_verb_metrics = self.tail_verb_evaluator.evaluate_verbose()
        else:
            tail_verb_metrics = {"top1_acc": "N/A", "top5_acc": "N/A"}
        if hasattr(self, "tail_action_evaluator_AND"):
            tail_action_metrics_AND = self.tail_action_evaluator_AND.evaluate_verbose()
        else:
            tail_action_metrics_AND = {"top1_acc": "N/A"}
        if hasattr(self, "tail_action_evaluator_OR"):
            tail_action_metrics_OR = self.tail_action_evaluator_OR.evaluate_verbose()
        else:
            tail_action_metrics_OR = {"top1_acc": "N/A"}

        return {
            "noun_acc": noun_metrics["top1_acc"],
            "verb_acc": verb_metrics["top1_acc"],
            "action_acc": action_metrics["top1_acc"],
            "tail_noun_acc": tail_noun_metrics["top1_acc"],
            "tail_verb_acc": tail_verb_metrics["top1_acc"],
            "tail_action_AND_acc": tail_action_metrics_AND["top1_acc"],
            "tail_action_OR_acc": tail_action_metrics_OR["top1_acc"],
        }

    def is_best(self):
        metrics = self.evaluate()
        # Validate whether it's the best model
        if metrics["action_acc"] > self.best_acc:
            self.best_acc = metrics["action_acc"]
            return True
        return False


evaluators_factory = {
    "something-something": EvaluatorSomething,
    "egtea-gaze": EvaluatorSomething,
    "EPIC-KITCHENS": EvaluatorEpic,
    "montalbano": EvaluatorSomething,
}
