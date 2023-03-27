from vrp_algorithms_lib.analytical_tools.routes_similarity.routes_similarity_metric_base import \
    RoutesSimilarityMetricBase
from vrp_algorithms_lib.problem.models import ProblemDescription
from vrp_algorithms_lib.problem.models import Routes


class ArcDifference(RoutesSimilarityMetricBase):
    """
    https://arxiv.org/pdf/2108.04578.pdf#cite.ceikute2013routing
    Arc Difference (AD) measures the number of arcs traveled in the actual solution but not
    in the MLE routing solution. It is calculated by taking the set difference of the arc sets of
    the test and predicted solutions. The percentage is computed by dividing AD by the total
    number of arcs in the whole routing.
    """
    def get_metric_name(self) -> str:
        return 'arc_difference'

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
