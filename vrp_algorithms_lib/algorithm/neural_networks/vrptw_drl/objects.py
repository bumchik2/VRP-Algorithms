from typing import Dict
from typing import List
from typing import Optional
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

    def get_filtered_partial_route(self) -> List[LocationId]:
        location_ids = []
        for i in range(len(self.partial_route) - 1):
            if self.partial_route[i + 1] != self.partial_route[i]:
                location_ids.append(self.partial_route[i + 1])
        return location_ids


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

    def get_delta_penalty(
            self,
            action: Action
    ) -> float:
        assert action.location_id in self.problem_description.locations

        max_penalty = 200
        if action.location_id in self.visited_location_ids:
            return max_penalty * 1.5  # Choosing incorrect location is the worst thing possible

        current_penalty = self.get_current_penalty()
        self.update(action)
        new_penalty = self.get_current_penalty()
        self._undo_action(action)
        delta_penalty = new_penalty - current_penalty
        return min(delta_penalty, max_penalty)

    def get_number_of_potential_latenesses(
            self,
            action: Action,
            routes: Routes
    ):
        # Get number of locations of the courier that have not been yet visited
        # and have lower time window begin than the chosen location
        courier_id = action.courier_id

        courier_routes = [route for route in routes.routes if route.vehicle_id == action.courier_id]
        assert len(courier_routes) == 1
        courier_route = courier_routes[0]

        partial_route = [vehicle_state for vehicle_state in self.vehicle_states
                         if vehicle_state.courier_id == courier_id][0].partial_route
        location_ids_to_visit = set(courier_route.location_ids) - set(partial_route)

        number_of_potential_latenesses = 0
        chosen_location_time_window_start_s = self.problem_description.locations[action.location_id].time_window_start_s

        for location_id in location_ids_to_visit:
            location = self.problem_description.locations[location_id]
            current_time_window_start_s = location.time_window_start_s
            if current_time_window_start_s < chosen_location_time_window_start_s:
                number_of_potential_latenesses += 1

        return number_of_potential_latenesses

    def get_lateness_risk_reward(
            self,
            action: Action,
            routes: Optional[Routes]
    ) -> float:
        if routes is None:
            return 0

        number_of_potential_latenesses = self.get_number_of_potential_latenesses(action, routes)

        potential_lateness_multiplier = 3.0
        max_potential_lateness_penalty = 100.0

        return -min(potential_lateness_multiplier * number_of_potential_latenesses, max_potential_lateness_penalty)

    def get_reward(
            self,
            action: Action,
            routes: Optional[Routes]
    ) -> float:
        delta_penalty = self.get_delta_penalty(action)
        lateness_risk_reward = self.get_lateness_risk_reward(action, routes)
        return -delta_penalty + lateness_risk_reward

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

    def get_vehicle_state_by_courier_id(
            self,
            courier_id: CourierId
    ) -> VehicleState:
        vehicle_state_wrapped = [vehicle_state for vehicle_state in self.vehicle_states
                                 if vehicle_state.courier_id == courier_id]
        assert len(vehicle_state_wrapped) == 1
        return vehicle_state_wrapped[0]


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
        location_ids: List[LocationId] = vehicle_state.get_filtered_partial_route()
        routes.append(Route(
            vehicle_id=vehicle_state.courier_id,
            location_ids=location_ids
        ))

    routes = Routes(routes=routes)
    return routes
