from vrp_algorithms_lib.problem.models import ProblemDescription, Routes, Penalties
from vrp_algorithms_lib.problem.penalties.base_penalty_calculator import BasePenaltyCalculator
from vrp_algorithms_lib.problem.metrics.global_proximity_distance_calculator import GlobalProximityDistanceCalculator


class GlobalProximityPenaltyCalculator(BasePenaltyCalculator):
    @staticmethod
    def get_penalty_name() -> str:
        return 'global_proximity_penalty'

    def calculate(
            self,
            problem_description: ProblemDescription,
            routes: Routes
    ) -> float:
        penalty_multipliers: Penalties = problem_description.penalties
        penalty = penalty_multipliers.distance_penalty_multiplier * penalty_multipliers.global_proximity_factor * \
            GlobalProximityDistanceCalculator().calculate(problem_description=problem_description, routes=routes)
        return penalty
