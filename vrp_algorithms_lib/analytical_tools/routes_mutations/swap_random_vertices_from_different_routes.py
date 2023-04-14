import random
from typing import Optional

import numpy as np

from vrp_algorithms_lib.analytical_tools.routes_mutations.routes_mutation_base import RoutesMutationBase
from vrp_algorithms_lib.problem.models import Routes, Route


class SwapRandomVerticesFromDifferentRoutes(RoutesMutationBase):
    def modify_routes(
            self,
            routes: Routes
    ) -> None:
        if len(routes.routes) < 2:
            return

        random_route_1_idx = None
        random_route_1: Optional[Route] = None
        while random_route_1 is None or len(random_route_1.location_ids) < 1:
            random_route_1_idx = np.random.randint(low=0, high=len(routes.routes))
            random_route_1 = routes.routes[random_route_1_idx]

        random_route_2: Optional[Route] = None
        while random_route_2 is None or len(random_route_2.location_ids) < 1:
            random_route_2_idx = np.random.randint(low=0, high=len(routes.routes))
            if random_route_2_idx == random_route_1_idx:
                continue
            random_route_2 = routes.routes[random_route_2_idx]

        i = random.sample(range(len(random_route_1.location_ids)), k=1)[0]
        j = random.sample(range(len(random_route_2.location_ids)), k=1)[0]

        random_route_1.location_ids[i], random_route_2.location_ids[j] = \
            random_route_2.location_ids[j], random_route_1.location_ids[i]
