from vrp_algorithms_lib.problem.models import ProblemDescription, Routes, Penalties
from vrp_algorithms_lib.problem.penalties.base_penalty_calculator import BasePenaltyCalculator


class GlobalProximityPenaltyCalculator(BasePenaltyCalculator):
    @staticmethod
    def get_penalty_name(
    ) -> str:
        return 'global_proximity_penalty'

    def calculate(
            self,
            problem_description: ProblemDescription,
            routes: Routes
    ) -> float:
        penalty = 0
        penalty_multipliers: Penalties = problem_description.penalties

        if penalty_multipliers.distance_penalty_multiplier > 0 and penalty_multipliers.global_proximity_factor > 0:
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

            penalty += penalty_multipliers.distance_penalty_multiplier * total_distance * \
                penalty_multipliers.global_proximity_factor

        return penalty
