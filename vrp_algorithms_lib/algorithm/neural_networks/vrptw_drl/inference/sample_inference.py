from typing import Optional

import numpy as np
import torch

from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.inference.sequential_inference import SequentialInference
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.models.model_base import ModelBase
from vrp_algorithms_lib.problem.models import ProblemDescription
from vrp_algorithms_lib.problem.models import Routes


class SampleInference(SequentialInference):
    def __init__(
            self,
            model: ModelBase,
            problem_description: ProblemDescription,
            routes: Optional[Routes],
            epsilon: float = 1
    ):
        super().__init__(
            model,
            problem_description,
            routes
        )

        self._epsilon = epsilon

    def set_epsilon(
            self,
            epsilon: float
    ):
        self._epsilon = epsilon

    def multiply_epsilon(
            self,
            multiplier: float
    ):
        self._epsilon *= multiplier

    def choose_courier_idx(
            self,
            courier_logits: torch.Tensor
    ) -> int:
        p = torch.nn.Softmax()(courier_logits).to(torch.device('cpu')).numpy()

        gen = np.random.uniform()
        if gen <= self._epsilon:
            return np.random.choice(range(len(courier_logits)), p=p)
        else:
            return np.argmax(p)

    def choose_location_idx(
            self,
            locations_logits: torch.Tensor
    ) -> int:
        p = torch.nn.Softmax()(locations_logits).to(torch.device('cpu')).numpy()
        return np.random.choice(range(len(locations_logits)), p=p)
