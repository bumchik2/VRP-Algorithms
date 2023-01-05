from copy import deepcopy
from typing import List
from typing import Optional

import numpy as np
import torch

from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.inference.base_inference import BaseInference
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.models.model_base import ModelBase
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.objects import Action, ProblemState
from vrp_algorithms_lib.problem.models import ProblemDescription
from vrp_algorithms_lib.problem.models import Routes


class BeamInference(BaseInference):
    def __init__(
            self,
            model: ModelBase,
            problem_description: ProblemDescription,
            routes: Optional[Routes],
            beam: int,
            use_courier_logits: bool
    ):
        super().__init__(
            model,
            problem_description,
            routes
        )

        self._problem_states: List[ProblemState] = [self.problem_state]
        self._cum_logits = [0]

        self.beam = beam
        self.use_courier_logits = use_courier_logits

    def _solve_problem(self):
        locations_number = len(self.problem_state.problem_description.locations)

        for i in range(locations_number):
            new_problem_states: List[ProblemState] = []
            new_cum_logits: List[float] = []
            potential_updates: List[dict] = []

            for j, problem_state in enumerate(self._problem_states):
                courier_logits = self.model.get_couriers_logits(problem_state)

                for courier_idx, courier_logit in enumerate(courier_logits):
                    courier_id = problem_state.idx_to_courier_id[courier_idx]
                    locations_logits = self.model.get_locations_logits(courier_id, problem_state)
                    for visited_location_id in problem_state.visited_location_ids:
                        visited_location_idx = problem_state.location_id_to_idx[visited_location_id]
                        locations_logits[visited_location_idx] -= torch.tensor(1e6, dtype=torch.float32)

                    best_locations_logits, best_locations_indices = torch.topk(locations_logits, k=self.beam)

                    for location_idx in best_locations_indices.numpy():
                        location_id = problem_state.idx_to_location_id[location_idx]
                        if location_id in problem_state.visited_location_ids:
                            continue

                        potential_cum_logit = self._cum_logits[j] + locations_logits[location_idx]
                        if self.use_courier_logits:
                            potential_cum_logit += courier_logits[courier_idx]
                        potential_updates.append({
                            'potential_cum_logit': potential_cum_logit,
                            'potential_previous_problem_state_idx': j,
                            'potential_action': Action(courier_id=courier_id, location_id=location_id)
                        })

            assert len(potential_updates) >= self.beam

            best_potential_updates = sorted(potential_updates, key=lambda x: x['potential_cum_logit'])[-self.beam:]
            for best_potential_update in best_potential_updates:
                new_problem_states.append(deepcopy(
                    self._problem_states[best_potential_update['potential_previous_problem_state_idx']]
                ))
                new_problem_states[-1].update(best_potential_update['potential_action'])
                new_cum_logits.append(best_potential_update['potential_cum_logit'])

            self._problem_states = new_problem_states
            self._cum_logits = new_cum_logits

        assert len(self._problem_states) == self.beam
        best_problem_state_idx = np.argmax(self._cum_logits)
        self.problem_state = self._problem_states[best_problem_state_idx]
        self._problem_states = []  # release memory
