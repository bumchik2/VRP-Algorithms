import numpy as np
import torch

from vrp_algorithms_lib.algorithm.neural_networks.hcvrp_drl.models.model_base import ModelBase
from vrp_algorithms_lib.algorithm.neural_networks.hcvrp_drl.objects import ProblemState, Action


class GreedyModel(ModelBase):
    def _get_couriers_logits(self, problem_state: ProblemState) -> torch.tensor:
        # Choose random courier
        couriers_number = len(problem_state.problem_description.couriers)
        result = -torch.ones(couriers_number) * 1000.0
        result[np.random.randint(low=0, high=couriers_number)] = 1.
        return result

    def _get_locations_logits(self, courier_idx: int, problem_state: ProblemState) -> torch.tensor:
        # Choose the closest location to the courier_id, that has not been visited yet
        min_delta_distance = 10 ** 9
        closest_location_idx = None

        for i, location_id in enumerate(problem_state.problem_description.locations):
            courier_id = list(problem_state.problem_description.couriers.keys())[courier_idx]
            delta_distance = problem_state.get_delta_distance(Action(courier_id=courier_id, location_id=location_id))
            if delta_distance < min_delta_distance and location_id not in problem_state.visited_location_idx:
                min_delta_distance = delta_distance
                closest_location_idx = i

        locations_number = len(problem_state.problem_description.locations)
        result = -torch.ones(locations_number) * 1000.0
        result[closest_location_idx] = 1.
        return result
