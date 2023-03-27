from typing import List

from vrp_algorithms_lib.analytical_tools.routes_similarity.arc_difference import ArcDifference
from vrp_algorithms_lib.analytical_tools.routes_similarity.route_difference import RouteDifference
from vrp_algorithms_lib.analytical_tools.routes_similarity.routes_similarity_metric_base import RoutesSimilarityMetricBase

ALL_ROUTES_SIMILARITY_METRICS: List[RoutesSimilarityMetricBase] = [
    ArcDifference(),
    RouteDifference()
]
