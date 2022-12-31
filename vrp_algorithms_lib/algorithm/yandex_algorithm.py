from vrp_algorithms_lib.algorithm.base_algorithm import BaseAlgorithm
from vrp_algorithms_lib.problem.models import ProblemDescription, Routes


class YandexAlgorithm(BaseAlgorithm):
    def solve_problem(
            self,
            problem_description: ProblemDescription
    ) -> Routes:
        # Not sure if I want to implement this.
        # It would require usage of some internal yandex libraries and
        # "problem description to yandex request" converter
        raise NotImplementedError
