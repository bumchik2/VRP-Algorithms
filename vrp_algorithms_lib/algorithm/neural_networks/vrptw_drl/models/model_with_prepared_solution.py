from typing import Optional, List

import torch

from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.models.common_utils import choose_next_location_id, \
    choose_next_courier_id
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.models.model_base import ModelBase
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.objects import Routes, Route, ProblemState, CourierId


class ModelWithPreparedSolution(ModelBase):
    def __init__(
            self,
            device: torch.device = torch.device('cpu')
    ):
        super().__init__(device)

        self.routes = None

    def initialize(
            self,
            problem_state: ProblemState,
            routes: Optional[Routes]
    ):
        assert routes is not None
        self.routes = routes

    def _check_problem_state_matches_routes(
            self,
            problem_state: ProblemState
    ):
        for vehicle_state in problem_state.vehicle_states:
            courier_id = vehicle_state.courier_id
            courier_routes: List[Route] = [route for route in self.routes.routes if route.vehicle_id == courier_id]

            if len(courier_routes) == 0:
                continue

            assert len(courier_routes) == 1
            planned_location_ids_for_vehicle = courier_routes[0].location_ids
            actual_location_ids_for_vehicle = vehicle_state.get_filtered_partial_route()

            assert len(actual_location_ids_for_vehicle) <= len(planned_location_ids_for_vehicle)
            for i in range(len(actual_location_ids_for_vehicle)):
                assert actual_location_ids_for_vehicle[i] == planned_location_ids_for_vehicle[i]

    def _get_couriers_logits(
            self,
            problem_state: ProblemState
    ) -> torch.tensor:
        # Choose the courier for which the route has not been chosen completely yet and whose next trip is shortest
        self._check_problem_state_matches_routes(problem_state)

        next_courier_id = choose_next_courier_id(problem_state=problem_state, routes=self.routes)
        chosen_courier_idx = problem_state.courier_id_to_idx[next_courier_id]

        couriers_number = len(problem_state.problem_description.couriers)
        result = -torch.ones(couriers_number) * 1000.0
        result[chosen_courier_idx] = 1.
        return result

    def _get_locations_logits(
            self,
            courier_id: CourierId,
            problem_state: ProblemState
    ) -> torch.tensor:
        # Choose the closest location to the courier_id, that has not been visited yet
        self._check_problem_state_matches_routes(problem_state)

        next_location_id = choose_next_location_id(
            problem_state=problem_state, routes=self.routes, next_courier_id=courier_id)
        next_location_idx = problem_state.location_id_to_idx[next_location_id]

        locations_number = len(problem_state.problem_description.locations)
        result = -torch.ones(locations_number) * 1000.0
        result[next_location_idx] = 1.
        return result
