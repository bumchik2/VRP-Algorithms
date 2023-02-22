from copy import deepcopy
from typing import Set, Dict

from vrp_algorithms_lib.algorithm.greedy_algorithm import GreedyAlgorithm
from vrp_algorithms_lib.problem.models import ProblemDescription
from vrp_algorithms_lib.problem.models import Route, LocationId, CourierId, Location


class GreedyAlgorithmV2(GreedyAlgorithm):
    def get_greedy_route(
            self,
            problem_description: ProblemDescription,
            location_ids_to_visit: Set[LocationId],
            courier_id: CourierId
    ) -> Route:
        location_ids_to_visit = deepcopy(location_ids_to_visit)

        route = Route(
            vehicle_id=courier_id,
            location_ids=[]
        )

        locations_to_visit: Dict[LocationId, Location] = {
            i: problem_description.locations[i] for i in location_ids_to_visit
        }

        previous_point_id = problem_description.get_depot().id

        while len(location_ids_to_visit) > 0:
            next_location = None
            min_distance = None

            min_tw = min([location.time_window_start_s for location in locations_to_visit.values()])
            location_ids_to_visit_with_min_tw = {
                i for i in locations_to_visit if locations_to_visit[i].time_window_start_s == min_tw
            }

            for potential_next_location_id in location_ids_to_visit_with_min_tw:
                potential_next_location = problem_description.locations[potential_next_location_id]

                if len(route.location_ids) == 0:
                    current_distance = problem_description.distance_matrix.depots_to_locations_distances[
                        previous_point_id][potential_next_location_id]
                else:
                    current_distance = problem_description.distance_matrix.locations_to_locations_distances[
                        previous_point_id][potential_next_location_id]

                if next_location is None or current_distance < min_distance:
                    next_location = potential_next_location
                    min_distance = current_distance

            route.location_ids.append(next_location.id)
            previous_point_id = next_location.id
            location_ids_to_visit.discard(next_location.id)
            del locations_to_visit[next_location.id]

        return route
