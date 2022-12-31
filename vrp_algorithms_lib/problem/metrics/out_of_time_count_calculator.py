import numpy as np

from vrp_algorithms_lib.problem.metrics.base_metric_calculator import BaseMetricCalculator
from vrp_algorithms_lib.problem.models import ProblemDescription, Routes, Penalties
from vrp_algorithms_lib.problem.visit_time_scheduler import VisitTimeScheduler


class OutOfTimeCountCalculator(BaseMetricCalculator):
    @staticmethod
    def get_metric_name() -> str:
        return 'out_of_time_count'

    def calculate(
            self,
            problem_description: ProblemDescription,
            routes: Routes
    ) -> int:
        total_out_of_time_count = 0

        penalty_multipliers: Penalties = problem_description.penalties

        if penalty_multipliers.out_of_time_penalty_per_minute > 0:
            for route in routes.routes:
                visit_times = VisitTimeScheduler.get_locations_visit_times(problem_description, route)

                for location_id, visit_time in zip(route.location_ids, visit_times):
                    location = problem_description.locations[location_id]
                    out_of_time_s: int = max(0, visit_time - location.time_window_end_s,
                                             location.time_window_start_s - visit_time)
                    out_of_time_minutes: float = out_of_time_s / 60.0

                    if out_of_time_minutes > 0 and not np.isclose(0, out_of_time_minutes):
                        total_out_of_time_count += 1

        return total_out_of_time_count
