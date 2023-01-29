from vrp_algorithms_lib.problem.models import ProblemDescription, Routes
from vrp_algorithms_lib.problem.metrics.base_metric_calculator import BaseMetricCalculator


class ProximityDistanceCalculator(BaseMetricCalculator):
    @staticmethod
    def get_metric_name() -> str:
        return 'proximity_distance'

    def calculate(
            self,
            problem_description: ProblemDescription,
            routes: Routes
    ) -> float:
        total_distance = 0
        distance_matrix = problem_description.distance_matrix
        depot_id = problem_description.get_depot().id

        for route in routes.routes:
            if len(route.location_ids) < 2:
                continue

            total_distance += distance_matrix.depots_to_locations_distances[depot_id][route.location_ids[1]]

            for i in range(0, len(route.location_ids) - 2):
                location_1 = route.location_ids[i]
                location_2 = route.location_ids[i + 2]
                total_distance += distance_matrix.locations_to_locations_distances[location_1][location_2]

        return total_distance
