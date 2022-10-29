from abc import ABC
from abc import abstractmethod

from vrp_algorithms_lib.problem.models import ProblemDescription
from vrp_algorithms_lib.problem.models import Routes


class BaseAlgorithm(ABC):
    @abstractmethod
    def solve_problem(self, problem_description: ProblemDescription) -> Routes:
        raise NotImplementedError('solve_problem is not implemented for BaseClusterizer')
