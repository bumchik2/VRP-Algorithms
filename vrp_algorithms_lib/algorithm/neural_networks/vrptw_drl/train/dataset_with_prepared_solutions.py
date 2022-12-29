import random
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import scipy.stats as sps
import torch.utils.data as data_utils

import vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.train.common_utils as common_utils
from vrp_algorithms_lib.problem.models import ProblemDescription, Routes


class DatasetWithPreparedSolutions(data_utils.Dataset):
    def __init__(
            self,
            problem_description_list: List[ProblemDescription],
            routes_list: List[Routes],
            num_vehicles: Optional[int] = None,
            min_vehicles: Optional[int] = None,
            max_vehicles: Optional[int] = None,
            dataset_size: int = 100
    ):
        super().__init__()

        self.problem_description_list = problem_description_list
        self.routes_list = routes_list

        common_utils.common_min_max_value_check(num_vehicles, min_vehicles, max_vehicles)

        self.num_vehicles = num_vehicles
        self.min_vehicles = min_vehicles
        self.max_vehicles = max_vehicles

        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def _get_random_problem_description_and_routes(
            self,
            num_vehicles: int,
    ) -> Tuple[ProblemDescription, Routes]:
        # Choose random problem description out of problem_description_list
        # Choose random num_vehicles routes out of there.
        random_problem_description_idx = np.random.randint(low=0, high=len(self.problem_description_list))
        chosen_problem_description = self.problem_description_list[random_problem_description_idx]
        chosen_routes = self.routes_list[random_problem_description_idx]

        # Only choose vehicles with non-empty routes
        chose_not_empty_routes = False
        chosen_vehicle_ids = None

        while not chose_not_empty_routes:
            chosen_vehicle_indexes = random.sample(range(len(chosen_problem_description.couriers)), num_vehicles)
            chosen_vehicle_ids = [courier_id for i, courier_id in enumerate(chosen_problem_description.couriers)
                                  if i in chosen_vehicle_indexes]

            chose_not_empty_routes = True
            potential_routes = common_utils.get_routes_slice(chosen_routes, chosen_vehicle_ids)

            if len(potential_routes.routes) != num_vehicles:
                chose_not_empty_routes = False
                continue

            for route in potential_routes.routes:
                if len(route.location_ids) == 0:
                    chose_not_empty_routes = False
                    break

        assert len(chosen_vehicle_ids) == num_vehicles
        assert len(chosen_vehicle_ids) == len(set(chosen_vehicle_ids))
        routes = common_utils.get_routes_slice(chosen_routes, chosen_vehicle_ids)

        chosen_location_ids = []
        for route in routes.routes:
            for location_id in route.location_ids:
                chosen_location_ids.append(location_id)

        problem_description = common_utils.get_problem_description_slice(chosen_problem_description,
                                                                         chosen_vehicle_ids, chosen_location_ids)

        return problem_description, routes

    def __getitem__(self, idx: int) -> Tuple[ProblemDescription, Routes]:
        num_vehicles = self.num_vehicles if self.num_vehicles is not None else sps.randint(
            self.min_vehicles, self.max_vehicles + 1).rvs()
        random_problem_description, routes = self._get_random_problem_description_and_routes(num_vehicles)
        return random_problem_description, routes
