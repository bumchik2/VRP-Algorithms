from vrp_algorithms_lib.problem.models import ProblemDescription, Routes, Penalties, DepotId, LocationId
from vrp_algorithms_lib.problem.penalties.base_penalty_calculator import BasePenaltyCalculator


class DistancePenaltyCalculator(BasePenaltyCalculator):
    def calculate(
            self,
            problem_description: ProblemDescription,
            routes: Routes
    ) -> float:
        penalty = 0
        penalty_multipliers: Penalties = problem_description.penalties

        if penalty_multipliers.distance_penalty_multiplier > 0:
            total_distance = 0
            for route in routes.routes:
                if len(route.location_ids) == 0:
                    continue

                depot_id: DepotId = list(problem_description.depots.keys())[0]
                first_location_id: LocationId = route.location_ids[0]
                total_distance += problem_description.distance_matrix.depots_to_locations_distances[depot_id][
                    first_location_id]

                for i in range(len(route.location_ids) - 1):
                    location_id_1: LocationId = route.location_ids[i]
                    location_id_2: LocationId = route.location_ids[i + 1]
                    total_distance += problem_description.distance_matrix.locations_to_locations_distances[
                        location_id_1][location_id_2]

            penalty += penalty_multipliers.distance_penalty_multiplier * total_distance

        return penalty
