from vrp_algorithms_lib.problem.models import ProblemDescription, Routes
from vrp_algorithms_lib.problem.penalties.distance_penalty_calculator import DistancePenaltyCalculator
from vrp_algorithms_lib.problem.penalties.global_proximity_penalty_calculator import GlobalProximityPenaltyCalculator
from vrp_algorithms_lib.problem.penalties.base_penalty_calculator import BasePenaltyCalculator
from typing import List


ALL_PENALTY_CALCULATORS: List[BasePenaltyCalculator] = [
    DistancePenaltyCalculator(),
    GlobalProximityPenaltyCalculator()
]


class TotalPenaltyCalculator(BasePenaltyCalculator):
    def calculate(
            self,
            problem_description: ProblemDescription,
            routes: Routes
    ) -> float:
        total_penalty = 0

        for penalty_calculator in ALL_PENALTY_CALCULATORS:
            total_penalty += penalty_calculator.calculate(problem_description, routes)

        return total_penalty
