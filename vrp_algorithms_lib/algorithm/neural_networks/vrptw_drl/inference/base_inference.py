from abc import abstractmethod
from typing import Optional

from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.models.model_base import ModelBase
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.objects import extract_routes_from_problem_state
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.objects import initialize_problem_state
from vrp_algorithms_lib.problem.models import ProblemDescription
from vrp_algorithms_lib.problem.models import Routes
import torch


class BaseInference:
    def __init__(
            self,
            model: ModelBase,
            problem_description: ProblemDescription,
            routes: Optional[Routes]
    ):
        self.model = model

        self.problem_state = initialize_problem_state(
            problem_description=problem_description,
        )

        self.model.initialize(self.problem_state, routes)

    @abstractmethod
    def _solve_problem(self):
        raise NotImplementedError

    @torch.no_grad()
    def solve_problem(self) -> Routes:
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()

        self._solve_problem()
        return extract_routes_from_problem_state(self.problem_state)
