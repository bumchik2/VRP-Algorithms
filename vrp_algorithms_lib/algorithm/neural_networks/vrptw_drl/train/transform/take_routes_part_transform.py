import random
from copy import deepcopy
from typing import Tuple

from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.train.common_utils import get_problem_description_slice
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.train.transform.transform import Transform
from vrp_algorithms_lib.problem.models import ProblemDescription, Routes
from typing import Optional


class TakeRoutesPartTransform(Transform):
    def __init__(
            self,
            locations_share: Optional[float]=None,
            max_locations_in_route: Optional[int]=None
    ):
        super().__init__()

        assert (max_locations_in_route is not None) or (locations_share is not None)

        if max_locations_in_route is not None:
            assert max_locations_in_route > 0

        if locations_share is not None:
            assert 0 <= locations_share <= 1

        self.locations_share = locations_share
        self.max_locations_in_route = max_locations_in_route

    def __call__(
            self,
            problem_description: ProblemDescription,
            routes: Routes
    ) -> Tuple[ProblemDescription, Routes]:
        problem_description = deepcopy(problem_description)
        routes = deepcopy(routes)

        all_location_ids_left = set()

        for route in routes.routes:
            locations_to_take_number = len(route.location_ids)

            if self.locations_share is not None:
                locations_to_take_number = round(len(route.location_ids) * self.locations_share)

            if self.max_locations_in_route is not None:
                locations_to_take_number = min(len(route.location_ids), self.max_locations_in_route)

            assert locations_to_take_number > 0
            locations_idx_to_leave = random.sample(range(len(route.location_ids)), k=locations_to_take_number)
            route.location_ids = [location_id for i, location_id in enumerate(route.location_ids) if
                                  i in locations_idx_to_leave]
            assert len(locations_idx_to_leave) == locations_to_take_number

            for location_id in route.location_ids:
                all_location_ids_left.add(location_id)

        problem_description = get_problem_description_slice(
            problem_description=problem_description,
            courier_ids=list(problem_description.couriers.keys()),
            location_ids=list(all_location_ids_left)
        )

        return problem_description, routes
