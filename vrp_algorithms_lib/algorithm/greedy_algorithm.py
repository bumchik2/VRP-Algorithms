from vrp_algorithms_lib.algorithm.base_algorithm import BaseAlgorithm
from vrp_algorithms_lib.algorithm.clusterization.clusterizer_name import CLUSTERIZER_NAME_TO_CLUSTERIZER_TYPE
from vrp_algorithms_lib.algorithm.clusterization.clusterizer_name import ClusterizerName
from vrp_algorithms_lib.problem.models import ProblemDescription
from vrp_algorithms_lib.problem.models import Route
from vrp_algorithms_lib.problem.models import Routes
from typing import List


class GreedyAlgorithm(BaseAlgorithm):
    def __init__(self, clusterizer_name: ClusterizerName.T):
        self.clusterizer = CLUSTERIZER_NAME_TO_CLUSTERIZER_TYPE[clusterizer_name]()

    def solve_problem(self, problem_description: ProblemDescription) -> Routes:
        assert len(problem_description.depots) == 1, 'Multiple depots not supported'

        routes = Routes(routes=[])

        location_points: List[List[float]] = [[location.lon, location.lat] for location in
                                              problem_description.locations.values()]
        clusters = self.clusterizer.clusterize(location_points, len(problem_description.couriers))

        for cluster_number, courier in enumerate(problem_description.couriers.values()):
            location_ids_to_visit = set(
                [location.id for current_cluster_number, location in
                 zip(clusters, problem_description.locations.values())
                 if current_cluster_number == cluster_number]
            )

            route = Route(
                vehicle_id=courier.id,
                location_ids=[]
            )

            previous_point_id = list(problem_description.depots.values())[0].id

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

            routes.routes.append(route)

        return routes
