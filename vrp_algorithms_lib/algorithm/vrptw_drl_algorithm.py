from typing import Type, Optional, Dict, Any

import torch

from vrp_algorithms_lib.algorithm.base_algorithm import BaseAlgorithm
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.inference.base_inference import BaseInference
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.models.model_base import ModelBase
from vrp_algorithms_lib.problem.models import ProblemDescription, Routes


class VrptwDrlAlgorithm(BaseAlgorithm):
    def __init__(
            self,
            model: ModelBase,
            inference_class,
            inference_params: Optional[Dict[str, Any]] = None
    ):
        self.model = model
        self.inference_class = inference_class
        self.inference_params = inference_params

    @torch.no_grad()
    def solve_problem(self, problem_description: ProblemDescription) -> Routes:
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()

        inference = self.inference_class(
            model=self.model,
            problem_description=problem_description,
            routes=None,
            **self.inference_params
        )

        routes = inference.solve_problem()
        return routes
