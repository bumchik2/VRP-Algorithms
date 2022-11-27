from abc import ABC, abstractmethod

import torch

from vrp_algorithms_lib.algorithm.neural_networks.hcvrp_drl.objects import ProblemState


class ModelBase(ABC):
    def __init__(self, device: str):
        self.device = device

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
            courier_idx: int,
            problem_state: ProblemState
    ) -> torch.tensor:
        raise NotImplementedError

    def get_locations_logits(
            self,
            courier_idx: int,
            problem_state: ProblemState
    ):
        return self._get_locations_logits(courier_idx, problem_state).to(self.device)
