from abc import ABC, abstractmethod

from vrp_algorithms_lib.problem.models import Routes, ProblemDescription


class RoutesSimilarityMetricBase(ABC):
    @abstractmethod
    def get_metric_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def calculate(
            self,
            routes_1: Routes,
            routes_2: Routes,
            problem_description: ProblemDescription
    ) -> float:
        raise NotImplementedError
