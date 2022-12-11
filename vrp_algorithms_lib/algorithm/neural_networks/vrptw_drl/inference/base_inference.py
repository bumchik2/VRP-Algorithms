from abc import abstractmethod

from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.models.model_base import ModelBase
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.objects import extract_routes_from_problem_state
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.objects import initialize_problem_state
from vrp_algorithms_lib.problem.models import ProblemDescription
from vrp_algorithms_lib.problem.models import Routes


class BaseInference:
    def __init__(
            self,
            model: ModelBase,
            problem_description: ProblemDescription,
            initial_capacity: int = 100,
            device: str = 'cpu'
    ):
        self.model = model

        self.problem_state = initialize_problem_state(
            problem_description=problem_description,
            initial_capacity=initial_capacity
        )
        self.device = device

    @abstractmethod
    def _solve_problem(self):
        raise NotImplementedError

    def solve_problem(self) -> Routes:
        self._solve_problem()
        return extract_routes_from_problem_state(self.problem_state)
