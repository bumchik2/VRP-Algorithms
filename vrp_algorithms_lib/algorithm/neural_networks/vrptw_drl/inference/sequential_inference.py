from abc import ABC, abstractmethod

import torch

from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.inference.base_inference import BaseInference
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.objects import Action


class SequentialInference(BaseInference, ABC):
    @abstractmethod
    def choose_courier_idx(
            self,
            courier_logits: torch.Tensor
    ) -> int:
        raise NotImplementedError

    @abstractmethod
    def choose_location_idx(
            self,
            locations_logits: torch.Tensor
    ) -> int:
        raise NotImplementedError

    def _solve_problem(self):
        locations_number = len(self.problem_state.problem_description.locations)

        for i in range(locations_number):
            # Choose the next courier greedily
            courier_logits = self.model.get_couriers_logits(self.problem_state)
            courier_idx = self.choose_courier_idx(courier_logits)
            courier_id = self.problem_state.idx_to_courier_id[courier_idx]

            # Choose the next location greedily
            locations_logits = self.model.get_locations_logits(courier_id, self.problem_state)

            for visited_location_id in self.problem_state.visited_location_ids:
                visited_location_idx = self.problem_state.location_id_to_idx[visited_location_id]
                locations_logits[visited_location_idx] -= torch.tensor(1e6, dtype=torch.float32)

            location_idx = self.choose_location_idx(locations_logits)
            location_id = self.problem_state.idx_to_location_id[location_idx]

            action = Action(courier_id=courier_id, location_id=location_id)
            self.problem_state.update(action)
