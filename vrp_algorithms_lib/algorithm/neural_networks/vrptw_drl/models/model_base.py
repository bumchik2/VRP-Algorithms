from abc import ABC, abstractmethod
from typing import Optional

import torch

from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.objects import CourierId
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.objects import ProblemState
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.objects import Routes


class ModelBase(ABC):
    def __init__(
            self,
            device: torch.device = torch.device('cpu')
    ):
        self.device = device

    @abstractmethod
    def initialize(
            self,
            problem_state: ProblemState,
            routes: Optional[Routes]
    ):
        raise NotImplementedError

    @abstractmethod
    def _get_couriers_logits(
            self,
            problem_state: ProblemState
    ) -> torch.tensor:
        raise NotImplementedError

    def get_couriers_logits(
            self,
            problem_state: ProblemState
    ):
        return self._get_couriers_logits(problem_state).to(self.device)

    @abstractmethod
    def _get_locations_logits(
            self,
            courier_id: CourierId,
            problem_state: ProblemState
    ) -> torch.tensor:
        raise NotImplementedError

    def get_locations_logits(
            self,
            courier_id: CourierId,
            problem_state: ProblemState
    ):
        return self._get_locations_logits(courier_id, problem_state).to(self.device)
