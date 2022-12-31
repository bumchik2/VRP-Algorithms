from abc import ABC
from typing import Union

from vrp_algorithms_lib.problem.models import ProblemDescription, Routes


class BaseMetricCalculator(ABC):
    @staticmethod
    def get_metric_name() -> str:
        raise NotImplementedError

    def calculate(
            self,
            problem_description: ProblemDescription,
            routes: Routes
    ) -> Union[int, float]:
        raise NotImplementedError
