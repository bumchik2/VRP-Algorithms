import numpy as np
import torch

from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.models.common_utils import choose_next_courier_id, \
    choose_next_location_id
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.models.model_base import ModelBase
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.objects import initialize_problem_state, Action
from vrp_algorithms_lib.problem.metrics.base_metric_calculator import BaseMetricCalculator
from vrp_algorithms_lib.problem.models import ProblemDescription, Routes


class ModelMetricCalculator(BaseMetricCalculator):
    def __init__(
            self,
            model: ModelBase,
            use_courier_logits: bool
    ):
        self.model = model
        self.use_courier_probabilities = use_courier_logits

    @staticmethod
    def get_metric_name(
    ) -> str:
        return 'model_penalty'

    @torch.no_grad()
    def calculate(
            self,
            problem_description: ProblemDescription,
            routes: Routes
    ) -> float:
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()

        total_locations_in_routes = sum([len(route.location_ids) for route in routes.routes])
        assert total_locations_in_routes == len(problem_description.locations)

        problem_state = initialize_problem_state(problem_description=problem_description)
        self.model.initialize(problem_state=problem_state, routes=None)

        model_metric = 0

        for i in range(total_locations_in_routes):
            next_courier_id = choose_next_courier_id(problem_state=problem_state, routes=routes)
            next_courier_idx = problem_state.courier_id_to_idx[next_courier_id]

            # Calculate logits for the next courier
            courier_logits = self.model.get_couriers_logits(problem_state)
            courier_probability = torch.nn.Softmax()(courier_logits)[next_courier_idx].to(torch.device('cpu')).item()
            courier_log_prob = np.log(courier_probability)
            if self.use_courier_probabilities:
                model_metric += courier_log_prob

            next_location_id = choose_next_location_id(
                problem_state=problem_state, routes=routes, next_courier_id=next_courier_id)
            next_location_idx = problem_state.location_id_to_idx[next_location_id]

            # Calculate logits for the next location
            locations_logits = self.model.get_locations_logits(problem_state=problem_state, courier_id=next_courier_id)
            location_probability = torch.nn.Softmax()(locations_logits)[
                next_location_idx].to(torch.device('cpu')).item()
            location_log_prob = np.log(location_probability)
            model_metric += location_log_prob

            # Update problem state
            action = Action(courier_id=next_courier_id, location_id=next_location_id)
            problem_state.update(action)

        return model_metric
