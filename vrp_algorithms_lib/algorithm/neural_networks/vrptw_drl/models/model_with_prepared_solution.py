from typing import Optional, List, Union

import torch

from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.models.model_base import ModelBase
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.objects import Routes, Route, LocationId, DepotId, \
    ProblemState, CourierId


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

        min_distance = 10 ** 9
        chosen_courier_id = None

        for courier_id in problem_state.problem_description.couriers:
            courier_routes: List[Route] = [route for route in self.routes.routes if route.vehicle_id == courier_id]
            assert len(courier_routes) == 1
            planned_location_ids_for_vehicle = courier_routes[0].location_ids
            actual_location_ids_for_vehicle = [vehicle_state for vehicle_state in problem_state.vehicle_states
                                               if vehicle_state.courier_id == courier_id][
                0].get_filtered_partial_route()
            if len(actual_location_ids_for_vehicle) == len(planned_location_ids_for_vehicle):
                continue

            next_location_id = planned_location_ids_for_vehicle[len(actual_location_ids_for_vehicle)]
            assert len(problem_state.problem_description.depots) == 1
            depot_id = list(problem_state.problem_description.depots)[0]
            last_location_id_in_route: Union[LocationId, DepotId] = actual_location_ids_for_vehicle[-1] \
                if len(actual_location_ids_for_vehicle) > 0 else depot_id
            distance_to_next_location = problem_state.problem_description.distance_matrix.depots_to_locations_distances[
                last_location_id_in_route][next_location_id] if len(actual_location_ids_for_vehicle) == 0 else \
                problem_state.problem_description.distance_matrix.locations_to_locations_distances[
                    last_location_id_in_route][next_location_id]

            if distance_to_next_location < min_distance:
                min_distance = distance_to_next_location
                chosen_courier_id = courier_id

        chosen_courier_idx = problem_state.courier_id_to_idx[chosen_courier_id]

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

        courier_routes: List[Route] = [route for route in self.routes.routes if route.vehicle_id == courier_id]
        assert len(courier_routes) == 1
        planned_location_ids_for_vehicle = courier_routes[0].location_ids
        actual_location_ids_for_vehicle = [vehicle_state for vehicle_state in problem_state.vehicle_states
                                           if vehicle_state.courier_id == courier_id][0].get_filtered_partial_route()

        next_location_id = planned_location_ids_for_vehicle[len(actual_location_ids_for_vehicle)]
        next_location_idx = problem_state.location_id_to_idx[next_location_id]

        locations_number = len(problem_state.problem_description.locations)
        result = -torch.ones(locations_number) * 1000.0
        result[next_location_idx] = 1.
        return result
