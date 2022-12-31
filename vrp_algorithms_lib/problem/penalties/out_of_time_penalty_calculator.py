from vrp_algorithms_lib.problem.metrics.out_of_time_calculator import OutOfTimeCalculator
from vrp_algorithms_lib.problem.models import ProblemDescription, Routes
from vrp_algorithms_lib.problem.penalties.base_penalty_calculator import BasePenaltyCalculator


class OutOfTimePenaltyCalculator(BasePenaltyCalculator):
    @staticmethod
    def get_penalty_name() -> str:
        return 'out_of_time_penalty'

    def calculate(
            self,
            problem_description: ProblemDescription,
            routes: Routes
    ) -> float:
        total_out_of_time_minutes = OutOfTimeCalculator().calculate(
            problem_description=problem_description, routes=routes)

        penalty = total_out_of_time_minutes * problem_description.penalties.out_of_time_penalty_per_minute

        return penalty
