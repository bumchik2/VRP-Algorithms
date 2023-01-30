from typing import List
from typing import Optional

import scipy.stats as sps
import torch

from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.models.attention_model import AttentionModel
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.objects import ProblemState
from vrp_algorithms_lib.common_tools.misc import get_euclidean_distance_km
from vrp_algorithms_lib.problem.models import Point, Routes


class AttentionModelWithEuclideanDistanceMatrix(AttentionModel):
    def __init__(
            self,
            number_of_pivot_points: int,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.number_of_pivot_points = number_of_pivot_points
        self.pivot_points = None

    def get_pivot_points(
            self,
            problem_state: ProblemState
    ) -> List[Point]:
        locations_and_depot_points = [location.point for location in
                                      problem_state.problem_description.locations.values()] + \
                                     [problem_state.problem_description.get_depot().point]

        min_lat = min([point.lat for point in locations_and_depot_points])
        max_lat = max([point.lat for point in locations_and_depot_points])
        min_lon = min([point.lon for point in locations_and_depot_points])
        max_lon = max([point.lon for point in locations_and_depot_points])

        lat_distribution = sps.uniform(loc=min_lat, scale=max_lat - min_lat)
        lon_distribution = sps.uniform(loc=min_lon, scale=max_lon - min_lon)

        lats = lat_distribution.rvs(self.number_of_pivot_points)
        lons = lon_distribution.rvs(self.number_of_pivot_points)
        result = [Point(lat=lats[i], lon=lons[i]) for i in range(self.number_of_pivot_points)]

        return result

    def initialize(
            self,
            problem_state: ProblemState,
            routes: Optional[Routes]
    ):
        super().initialize(problem_state=problem_state, routes=routes)
        self.pivot_points = self.get_pivot_points(problem_state=problem_state)

    def get_euclidean_distances_to_pivot_points(
            self,
            problem_state: ProblemState
    ) -> torch.Tensor:
        euclidean_distances_to_pivot_points = []
        locations_and_depot_points = [location.point for location in
                                      problem_state.problem_description.locations.values()] + \
                                     [problem_state.problem_description.get_depot().point]

        for point in locations_and_depot_points:
            euclidean_distances_to_pivot_points_part = []
            for pivot_point in self.pivot_points:
                euclidean_distances_to_pivot_points_part.append(
                    get_euclidean_distance_km(point.lat, point.lon, pivot_point.lat, pivot_point.lon))
            euclidean_distances_to_pivot_points.append(euclidean_distances_to_pivot_points_part)

        return torch.tensor(euclidean_distances_to_pivot_points).to(self.device)

    def get_locations_information(
            self,
            problem_state: ProblemState
    ) -> torch.Tensor:
        locations_information_part = super().get_locations_information(problem_state)

        euclidean_distances_to_pivot_points = self.get_euclidean_distances_to_pivot_points(
            problem_state=problem_state
        )

        return torch.concat(
            [locations_information_part, euclidean_distances_to_pivot_points * 0.01],  # 0.01 is normalizing constant
            dim=1
        )  # (n + 1) x (dim_1 + dim_2)
