import torch

from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.inference.base_inference import BaseInference
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.objects import Action


class GreedyInference(BaseInference):
    def _solve_problem(self):
        locations_number = len(self.problem_state.problem_description.locations)

        for i in range(locations_number):
            # Choose the next courier greedily
            courier_logits = self.model.get_couriers_logits(self.problem_state)
            courier_idx = torch.argmax(courier_logits).item()
            courier_id = list(self.problem_state.problem_description.couriers)[courier_idx]

            # Choose the next location greedily
            locations_logits = self.model.get_locations_logits(courier_idx, self.problem_state)
            location_idx = torch.argmax(locations_logits).item()
            location_id = list(self.problem_state.problem_description.locations)[location_idx]

            action = Action(courier_id=courier_id, location_id=location_id)
            self.problem_state.update(action)
