import random

import numpy as np

from vrp_algorithms_lib.analytical_tools.routes_mutations.routes_mutation_base import RoutesMutationBase
from vrp_algorithms_lib.problem.models import Routes


class SwapRandomNeighbouringVerticesWithinRoute(RoutesMutationBase):
    def modify_routes(
            self,
            routes: Routes
    ) -> None:
        random_route = routes.routes[np.random.randint(low=0, high=len(routes.routes))]

        i = random.choice(range(len(random_route.location_ids) - 1))
        j = i + 1

        random_route.location_ids[i], random_route.location_ids[j] = \
            random_route.location_ids[j], random_route.location_ids[i]
