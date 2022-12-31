from vrp_algorithms_lib.problem.metrics.base_metric_calculator import BaseMetricCalculator
from vrp_algorithms_lib.problem.models import ProblemDescription, Routes


class DistanceCalculator(BaseMetricCalculator):
    @staticmethod
    def get_metric_name() -> str:
        return 'distance_km'

    def calculate(
            self,
            problem_description: ProblemDescription,
            routes: Routes
    ) -> float:
        total_distance_km = 0

        for route in routes.routes:
            if len(route.location_ids) == 0:
                continue

            depot_id = list(problem_description.depots.keys())[0]
            first_location_id = route.location_ids[0]
            total_distance_km += problem_description.distance_matrix.depots_to_locations_distances[depot_id][
                first_location_id]

            for i in range(len(route.location_ids) - 1):
                location_id_1 = route.location_ids[i]
                location_id_2 = route.location_ids[i + 1]
                total_distance_km += problem_description.distance_matrix.locations_to_locations_distances[
                    location_id_1][location_id_2]

        return total_distance_km
