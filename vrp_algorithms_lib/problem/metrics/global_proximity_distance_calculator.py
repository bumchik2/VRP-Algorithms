from vrp_algorithms_lib.problem.metrics.base_metric_calculator import BaseMetricCalculator
from vrp_algorithms_lib.problem.models import ProblemDescription, Routes


class GlobalProximityDistanceCalculator(BaseMetricCalculator):
    @staticmethod
    def get_metric_name() -> str:
        return 'global_proximity_distance'

    def calculate(
            self,
            problem_description: ProblemDescription,
            routes: Routes
    ) -> float:
        total_distance = 0
        for route in routes.routes:
            if len(route.location_ids) == 0:
                continue

            last_location_id = route.location_ids[-1]
            distance_to_last_location = 0

            for location_id in route.location_ids[:-1]:
                distance_to_last_location += problem_description.distance_matrix.locations_to_locations_distances[
                    location_id][last_location_id]

            total_distance += distance_to_last_location
        return total_distance
