from copy import deepcopy
from typing import List, Set

from vrp_algorithms_lib.algorithm.base_algorithm import BaseAlgorithm
from vrp_algorithms_lib.algorithm.clusterization.clusterizer_name import CLUSTERIZER_NAME_TO_CLUSTERIZER_TYPE
from vrp_algorithms_lib.algorithm.clusterization.clusterizer_name import ClusterizerName
from vrp_algorithms_lib.problem.models import ProblemDescription
from vrp_algorithms_lib.problem.models import Route, LocationId, CourierId
from vrp_algorithms_lib.problem.models import Routes
from vrp_algorithms_lib.problem.penalties.total_penalty_calculator import TotalPenaltyCalculator


class GreedyAlgorithm(BaseAlgorithm):
    def __init__(
            self,
            clusterizer_name: ClusterizerName.T
    ):
        self.clusterizer = CLUSTERIZER_NAME_TO_CLUSTERIZER_TYPE[clusterizer_name]()

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

        previous_point_id = problem_description.get_depot().id

        while len(location_ids_to_visit) > 0:
            next_location = None
            min_distance = None

            for potential_next_location_id in location_ids_to_visit:
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

        return route

    def solve_problem(
            self,
            problem_description: ProblemDescription
    ) -> Routes:
        min_penalty = 10 ** 9
        best_routes = None

        assert len(problem_description.depots) == 1, 'Multiple depots not supported'

        location_points: List[List[float]] = [[location.point.lon, location.point.lat] for location in
                                              problem_description.locations.values()]

        for number_of_clusters in range(1, len(problem_description.couriers) + 1):
            routes = Routes(routes=[])

            clusters = self.clusterizer.clusterize(location_points, number_of_clusters=number_of_clusters)

            for cluster_number in range(number_of_clusters):
                courier_id = list(problem_description.couriers)[cluster_number]
                location_ids_to_visit = set(
                    [location.id for current_cluster_number, location in
                     zip(clusters, problem_description.locations.values())
                     if current_cluster_number == cluster_number]
                )

                route = self.get_greedy_route(
                    problem_description,
                    location_ids_to_visit,
                    courier_id
                )

                routes.routes.append(route)

            assert (len(routes.routes) == number_of_clusters)

            total_penalty = TotalPenaltyCalculator().calculate(
                problem_description=problem_description, routes=routes)

            if total_penalty < min_penalty:
                min_penalty = total_penalty
                best_routes = deepcopy(routes)

        return best_routes
