from vrp_algorithms_lib.analytical_tools.routes_similarity.routes_similarity_metric_base import RoutesSimilarityMetricBase
from vrp_algorithms_lib.problem.models import ProblemDescription
from vrp_algorithms_lib.problem.models import Routes


class RouteDifference(RoutesSimilarityMetricBase):
    """
    https://arxiv.org/pdf/2108.04578.pdf#cite.ceikute2013routing
    Route Difference (RD) indicates the number of stops that were incorrectly assigned to
    a different route. Intuitively, RD may be interpreted as an estimate of how many moves
    between routes are necessary when modifying the predicted MLE solution to match the
    actual routing. To compute RD, the pair of routes with the smallest difference in stops is
    greedily selected without replacement. The total number of incorrectly assigned stops is
    considered as RD. The percentage is computed by dividing RD by the total number of stops
    in the whole routing.
    """
    def get_metric_name(self) -> str:
        return 'route_difference'

    def calculate(
            self,
            routes_1: Routes,
            routes_2: Routes,
            problem_description: ProblemDescription
    ) -> float:
        used_vehicle_ids = set()

        incorrect_locations = 0
        total_locations = 0

        for route_1 in routes_1.routes:
            if len(route_1.location_ids) == 0:
                continue

            most_similar_route = None
            max_similarity = -1

            # Find a route that is the most similar to route_1
            for route_2 in routes_2.routes:
                if route_2.vehicle_id in used_vehicle_ids:
                    continue
                if len(route_2.location_ids) == 0:
                    continue

                locations_1 = set(route_1.location_ids)
                locations_2 = set(route_2.location_ids)
                common_locations = locations_1 & locations_2
                similarity = len(common_locations) / max(len(locations_1), len(locations_2))

                if similarity > max_similarity:
                    most_similar_route = route_2
                    max_similarity = similarity

            used_vehicle_ids.update(most_similar_route.vehicle_id)

            # Now for the route route_1 we have the most similar route_2
            # We can now calculate the number of locations in route_1 that were not assigned to route_2
            if most_similar_route is None:
                incorrect_locations_for_route = len(route_1.location_ids)
            else:
                incorrect_locations_for_route = len(set(route_1.location_ids) - set(most_similar_route.location_ids))

            incorrect_locations += incorrect_locations_for_route
            total_locations += len(route_1.location_ids)

        return incorrect_locations / total_locations
