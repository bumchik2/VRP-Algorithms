from vrp_algorithms_lib.problem.metrics.distance_calculator import DistanceCalculator
from vrp_algorithms_lib.problem.models import ProblemDescription, Routes, Penalties
from vrp_algorithms_lib.problem.penalties.base_penalty_calculator import BasePenaltyCalculator


class DistancePenaltyCalculator(BasePenaltyCalculator):
    @staticmethod
    def get_penalty_name() -> str:
        return 'distance_penalty'

    def calculate(
            self,
            problem_description: ProblemDescription,
            routes: Routes
    ) -> float:
        penalty = 0
        penalty_multipliers: Penalties = problem_description.penalties

        if penalty_multipliers.distance_penalty_multiplier > 0:
            total_distance = DistanceCalculator().calculate(problem_description=problem_description, routes=routes)
            penalty += penalty_multipliers.distance_penalty_multiplier * total_distance

        return penalty
