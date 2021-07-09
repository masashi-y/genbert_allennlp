from typing import NamedTuple

from allennlp.training.metrics.metric import Metric
from overrides import overrides


class CountResult(NamedTuple):
    total: float
    average: float


@Metric.register("counter")
class Counter(Metric):
    def __init__(self) -> None:
        self._total_value = 0.0
        self._count = 0

    @overrides
    def __call__(self, value):
        """
        # Parameters

        value : `float`
            The value to average.
        """
        self._total_value += list(self.detach_tensors(value))[0]
        self._count += 1

    @overrides
    def get_metric(self, reset: bool = False):
        """
        # Returns

        The average of all values that were passed to `__call__`.
        """
        average_value = self._total_value / self._count if self._count > 0 else 0
        total_value = self._total_value
        if reset:
            self.reset()
        return CountResult(total=total_value, average=average_value)

    @overrides
    def reset(self):
        self._total_value = 0.0
        self._count = 0
