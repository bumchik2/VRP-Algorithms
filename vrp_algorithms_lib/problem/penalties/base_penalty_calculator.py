from abc import ABC, abstractmethod

from vrp_algorithms_lib.problem.models import ProblemDescription, Routes


class BasePenaltyCalculator(ABC):
    @staticmethod
    def get_penalty_name() -> str:
        raise NotImplementedError

    @abstractmethod
    def calculate(
            self,
            problem_description: ProblemDescription,
            routes: Routes
    ) -> float:
        raise NotImplementedError
