from typing import Dict

import torch
from allennlp.training.metrics import Average, BooleanAccuracy, F1Measure
from allennlp_models.rc.metrics.drop_em_and_f1 import DropEmAndF1

from genbert.metrics.counter import Counter


class Metrics(object):
    def __init__(self):
        self.drop = DropEmAndF1()
        self._arg_f1 = F1Measure(positive_label=1)
        self._arg_f1_is_used = False
        self.type_accuracy = BooleanAccuracy()
        self._mean_metrics = {}
        self._count_metrics = {}
        self._loss_metrics = {}

    def arg_f1(self, preds, labels, mask):
        self._arg_f1_is_used = True
        preds = preds.float()
        return self._arg_f1(
            torch.stack([1 - preds, preds], dim=-1).float(),
            labels.float(),
            mask,
        )

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self.drop.get_metric(reset)
        answer_acc = self.type_accuracy.get_metric(reset)
        if self._arg_f1_is_used:
            arg_f1 = self._arg_f1.get_metric(reset)
        else:
            arg_f1 = {"f1": 0.0}

        results = {
            "type_accuracy": answer_acc,
            "per_instance_em": exact_match,
            "per_instance_f1": f1_score,
            "argument_f1": arg_f1["f1"],
        }

        for key, metric in self._mean_metrics.items():
            results[f"{key}_mean"] = float(metric.get_metric(reset))

        for key, metric in self._count_metrics.items():
            results[f"{key}_count"] = float(metric.get_metric(reset).total)

        for key, metric in self._loss_metrics.items():
            results[f"{key}_loss"] = float(metric.get_metric(reset))
        return results

    def mean_metrics(self, name: str, value: torch.tensor) -> None:
        if name not in self._mean_metrics:
            self._mean_metrics[name] = Average()
        self._mean_metrics[name](value)

    def loss_metrics(self, name: str, value: torch.tensor) -> None:
        if name not in self._loss_metrics:
            self._loss_metrics[name] = Average()
        self._loss_metrics[name](value)

    def count_metrics(self, name: str, value: torch.tensor) -> None:
        if name not in self._count_metrics:
            self._count_metrics[name] = Counter()
        self._count_metrics[name](value)
