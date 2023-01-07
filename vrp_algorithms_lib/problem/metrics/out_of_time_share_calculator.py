from vrp_algorithms_lib.problem.metrics.base_metric_calculator import BaseMetricCalculator
from vrp_algorithms_lib.problem.metrics.out_of_time_count_calculator import OutOfTimeCountCalculator
from vrp_algorithms_lib.problem.models import ProblemDescription, Routes


class OutOfTimeShareCalculator(BaseMetricCalculator):
    @staticmethod
    def get_metric_name() -> str:
        return 'out_of_time_share'

    def calculate(
            self,
            problem_description: ProblemDescription,
            routes: Routes
    ) -> float:
        total_out_of_time_count = OutOfTimeCountCalculator().calculate(
            problem_description=problem_description,
            routes=routes
        )

        return total_out_of_time_count / len(problem_description.locations)
