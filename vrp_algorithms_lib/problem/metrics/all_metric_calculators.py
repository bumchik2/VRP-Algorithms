from typing import List

from vrp_algorithms_lib.problem.metrics.base_metric_calculator import BaseMetricCalculator
from vrp_algorithms_lib.problem.metrics.distance_calculator import DistanceCalculator
from vrp_algorithms_lib.problem.metrics.global_proximity_distance_calculator import GlobalProximityDistanceCalculator
from vrp_algorithms_lib.problem.metrics.out_of_time_calculator import OutOfTimeCalculator
from vrp_algorithms_lib.problem.metrics.out_of_time_count_calculator import OutOfTimeCountCalculator
from vrp_algorithms_lib.problem.metrics.out_of_time_share_calculator import OutOfTimeShareCalculator
from vrp_algorithms_lib.problem.metrics.proximity_distance_calculator import ProximityDistanceCalculator

ALL_METRIC_CALCULATORS: List[BaseMetricCalculator] = [
    DistanceCalculator(),
    OutOfTimeCalculator(),
    OutOfTimeCountCalculator(),
    OutOfTimeShareCalculator(),
    ProximityDistanceCalculator(),
    GlobalProximityDistanceCalculator()
]
