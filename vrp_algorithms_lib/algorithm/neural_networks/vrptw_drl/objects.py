from typing import Dict
from typing import List
from typing import Set
from typing import Union

from pydantic import BaseModel

from vrp_algorithms_lib.problem.models import CourierId
from vrp_algorithms_lib.problem.models import DepotId
from vrp_algorithms_lib.problem.models import LocationId
from vrp_algorithms_lib.problem.models import ProblemDescription
from vrp_algorithms_lib.problem.models import Route
from vrp_algorithms_lib.problem.models import Routes
from vrp_algorithms_lib.problem.penalties.total_penalty_calculator import TotalPenaltyCalculator


class VehicleState(BaseModel):
    courier_id: CourierId
    total_distance: float
    partial_route: List[Union[DepotId, LocationId]]


def initialize_vehicle_state(
        courier_id: CourierId,
        depot_id: DepotId
) -> VehicleState:
    return VehicleState.parse_obj({
        'courier_id': courier_id,
        'total_distance': 0,  # distance in km
        'partial_route': [depot_id]
    })


class Action(BaseModel):
    courier_id: CourierId
    location_id: LocationId


class ProblemState(BaseModel):
    vehicle_states: List[VehicleState]
    problem_description: ProblemDescription
    visited_location_ids: Set[LocationId] = set()

    locations_idx: List[List[int]]
    location_id_to_idx: Dict[LocationId, int]
    idx_to_location_id: Dict[int, LocationId]
    courier_id_to_idx: Dict[CourierId, int]
    idx_to_courier_id: Dict[int, CourierId]

    def get_delta_distance(self, action) -> float:
        for vehicle_state in self.vehicle_states:
            if action.courier_id == vehicle_state.courier_id:
                distance_matrix = self.problem_description.distance_matrix
                if vehicle_state.partial_route[-1] == list(self.problem_description.depots.keys())[0]:
                    delta_distance = distance_matrix.depots_to_locations_distances[
                        vehicle_state.partial_route[-1]][action.location_id]
                else:
                    delta_distance = distance_matrix.locations_to_locations_distances[
                        vehicle_state.partial_route[-1]][action.location_id]
                return delta_distance
        raise ValueError(f'Unexpected Error in get_delta_distance: courier with id {action.courier_id} not found')

    def _undo_action(self, action: Action):
        assert action.location_id in self.visited_location_ids, f'{action.location_id} was not visited before'
        self.visited_location_ids.remove(action.location_id)

        for vehicle_state in self.vehicle_states:
            current_courier_idx = self.courier_id_to_idx[vehicle_state.courier_id]
            vehicle_state.partial_route.pop()
            self.locations_idx[current_courier_idx].pop()

        for vehicle_state in self.vehicle_states:
            if action.courier_id == vehicle_state.courier_id:
                delta_distance = self.get_delta_distance(action)
                vehicle_state.total_distance -= delta_distance

    def get_current_penalty(self):
        return TotalPenaltyCalculator().calculate(self.problem_description, extract_routes_from_problem_state(self))

    def get_reward(self, action: Action):
        assert action.location_id in self.problem_description.locations

        max_penalty = 100
        if action.location_id in self.visited_location_ids:
            return -max_penalty

        current_penalty = self.get_current_penalty()
        self.update(action)
        new_penalty = self.get_current_penalty()
        self._undo_action(action)
        delta_penalty = new_penalty - current_penalty

        return -min(delta_penalty, max_penalty)

    def update(self, action: Action):
        assert action.location_id not in self.visited_location_ids, f'{action.location_id} was visited before'
        self.visited_location_ids.update({action.location_id})
        new_location_idx = self.location_id_to_idx[action.location_id]

        for vehicle_state in self.vehicle_states:
            current_courier_idx = self.courier_id_to_idx[vehicle_state.courier_id]

            if action.courier_id == vehicle_state.courier_id:
                delta_distance = self.get_delta_distance(action)
                vehicle_state.total_distance += delta_distance
                vehicle_state.partial_route.append(action.location_id)
                self.locations_idx[current_courier_idx].append(new_location_idx)
            else:
                # All the routes should be of equal length at each step
                last_location_id_in_the_route = vehicle_state.partial_route[-1]
                vehicle_state.partial_route.append(last_location_id_in_the_route)
                self.locations_idx[current_courier_idx].append(len(self.problem_description.locations))


def initialize_problem_state(problem_description: ProblemDescription) -> ProblemState:
    locations_idx = [[len(problem_description.locations)] for _ in range(len(problem_description.couriers))]

    location_id_to_idx = {location.id: i for i, location in enumerate(problem_description.locations.values())}
    idx_to_location_id = {i: location_id for location_id, i in location_id_to_idx.items()}

    courier_id_to_idx = {courier.id: i for i, courier in enumerate(problem_description.couriers.values())}
    idx_to_courier_id = {i: courier_id for courier_id, i in courier_id_to_idx.items()}

    assert len(problem_description.depots) == 1, 'Multiple depots are not supported yet'
    depot_id = list(problem_description.depots.keys())[0]

    vehicle_states = [
        initialize_vehicle_state(
            courier_id=idx_to_courier_id[courier_idx],
            depot_id=depot_id
        ) for courier_idx in range(len(problem_description.couriers))
    ]

    return ProblemState(
        problem_description=problem_description,
        vehicle_states=vehicle_states,
        visited_location_ids=set(),
        locations_idx=locations_idx,
        location_id_to_idx=location_id_to_idx,
        idx_to_location_id=idx_to_location_id,
        courier_id_to_idx=courier_id_to_idx,
        idx_to_courier_id=idx_to_courier_id
    )


def extract_routes_from_problem_state(problem_state: ProblemState) -> Routes:
    routes = []

    for vehicle_state in problem_state.vehicle_states:
        location_ids: List[LocationId] = []
        route = vehicle_state.partial_route
        for i in range(len(route) - 1):
            if route[i + 1] != route[i]:
                location_ids.append(route[i + 1])

        routes.append(Route(
            vehicle_id=vehicle_state.courier_id,
            location_ids=location_ids
        ))

    routes = Routes(routes=routes)
    return routes
