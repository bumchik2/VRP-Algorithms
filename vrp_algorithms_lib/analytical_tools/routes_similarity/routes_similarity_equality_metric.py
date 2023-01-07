from vrp_algorithms_lib.analytical_tools.routes_similarity.routes_similarity_metric_base import \
    RoutesSimilarityMetricBase
from vrp_algorithms_lib.problem.models import Routes, ProblemDescription


class RoutesSimilarityEqualityMetric(RoutesSimilarityMetricBase):
    def calculate(
            self,
            routes_1: Routes,
            routes_2: Routes,
            problem_description: ProblemDescription
    ) -> float:
        edges_1 = routes_1.get_edges_set(problem_description.get_depot().id)
        edges_2 = routes_2.get_edges_set(problem_description.get_depot().id)
        assert len(edges_1) == len(edges_2)

        similarity_metric = 0
        for edge in edges_1:
            if edge in edges_2:
                similarity_metric += 1
        similarity_metric /= len(edges_1)

        return similarity_metric
