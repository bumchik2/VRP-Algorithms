from typing import List
from typing import Union
from typing import Set

from pydantic import BaseModel

from vrp_algorithms_lib.problem.models import CourierId
from vrp_algorithms_lib.problem.models import DepotId
from vrp_algorithms_lib.problem.models import LocationId
from vrp_algorithms_lib.problem.models import ProblemDescription
from vrp_algorithms_lib.problem.models import Route
from vrp_algorithms_lib.problem.models import Routes


class VehicleState(BaseModel):
    courier_id: CourierId
    remaining_capacity: int
    total_distance: float
    partial_route: List[Union[DepotId, LocationId]]


def initialize_vehicle_state(
        courier_id: CourierId,
        initial_capacity: int,
        depot_id: DepotId
) -> VehicleState:
    return VehicleState.parse_obj({
        'courier_id': courier_id,
        'remaining_capacity': initial_capacity,
        'total_distance': 0,
        'partial_route': [depot_id]
    })


class Action(BaseModel):
    courier_id: CourierId
    location_id: LocationId


class ProblemState(BaseModel):
    vehicle_states: List[VehicleState]
    problem_description: ProblemDescription
    visited_location_idx: Set[LocationId] = set()

    def get_delta_distance(self, action) -> float:
        for vehicle_state in self.vehicle_states:
            if action.courier_id == vehicle_state.courier_id:
                distance_matrix = self.problem_description.distance_matrix
                if vehicle_state.total_distance == 0:
                    delta_distance = distance_matrix.depots_to_locations_distances[
                        vehicle_state.partial_route[-1]][action.location_id]
                else:
                    delta_distance = distance_matrix.locations_to_locations_distances[
                        vehicle_state.partial_route[-1]][action.location_id]
                return delta_distance
        raise ValueError(f'Unexpected Error in get_delta_distance: courier with id {action.courier_id} not found')

    def get_reward(self, action: Action):
        delta_distance = self.get_delta_distance(action)
        return -delta_distance * self.problem_description.penalties.distance_penalty_multiplier

    def update(self, action: Action):
        assert action.location_id not in self.visited_location_idx, f'{action.location_id} was visited before'
        self.visited_location_idx.update({action.location_id})

        for vehicle_state in self.vehicle_states:
            if action.courier_id == vehicle_state.courier_id:
                delta_distance = self.get_delta_distance(action)
                vehicle_state.total_distance += delta_distance
                vehicle_state.remaining_capacity -= 1
                vehicle_state.partial_route.append(action.location_id)
            else:
                # All the routes should be of equal length at each step
                vehicle_state.partial_route.append(vehicle_state.partial_route[-1])


def initialize_problem_state(problem_description: ProblemDescription, initial_capacity: int) -> ProblemState:
    assert len(problem_description.depots) == 1, 'Multiple depots are not supported yet'
    depot_id = list(problem_description.depots.keys())[0]

    vehicle_states = [
        initialize_vehicle_state(
            courier_id=courier_id,
            initial_capacity=initial_capacity,
            depot_id=depot_id
        ) for courier_id in problem_description.couriers
    ]

    return ProblemState(
        problem_description=problem_description,
        vehicle_states=vehicle_states
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
