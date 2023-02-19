import random
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import scipy.stats as sps
import torch.utils.data as data_utils

import vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.train.common_utils as common_utils
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.train.transform.transform import Transform
from vrp_algorithms_lib.problem.models import ProblemDescription, Routes


class DatasetWithFixedPreparedSolutions(data_utils.Dataset):
    def __init__(
            self,
            problem_description_list: List[ProblemDescription],
            routes_list: List[Routes],
    ):
        super().__init__()

        assert len(problem_description_list) == len(routes_list)

        self.problem_description_list = problem_description_list
        self.routes_list = routes_list

        self.dataset_size = len(problem_description_list)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx: int) -> Tuple[ProblemDescription, Routes]:
        return self.problem_description_list[idx], self.routes_list[idx]


class DatasetWithPreparedSolutions(data_utils.Dataset):
    def __init__(
            self,
            problem_description_list: List[ProblemDescription],
            routes_list: List[Routes],
            num_vehicles: Optional[int] = None,
            min_vehicles: Optional[int] = None,
            max_vehicles: Optional[int] = None,
            dataset_size: int = 100,
            transform: Optional[Transform] = None
    ):
        super().__init__()

        self.problem_description_list = problem_description_list
        self.routes_list = routes_list

        common_utils.common_min_max_value_check(num_vehicles, min_vehicles, max_vehicles)

        self.num_vehicles = num_vehicles
        self.min_vehicles = min_vehicles
        self.max_vehicles = max_vehicles

        self.dataset_size = dataset_size

        self.transform = transform

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
        non_empty_couriers_ids = []
        for route in chosen_routes.routes:
            if len(route.location_ids) > 0:
                non_empty_couriers_ids.append(route.vehicle_id)

        chosen_vehicle_ids = random.sample(non_empty_couriers_ids, k=num_vehicles)
        assert len(chosen_vehicle_ids) == num_vehicles
        assert len(chosen_vehicle_ids) == len(set(chosen_vehicle_ids))
        routes = common_utils.get_routes_slice(chosen_routes, chosen_vehicle_ids)

        chosen_location_ids = []
        for route in routes.routes:
            assert len(route.location_ids) > 0
            for location_id in route.location_ids:
                chosen_location_ids.append(location_id)

        problem_description = common_utils.get_problem_description_slice(chosen_problem_description,
                                                                         chosen_vehicle_ids, chosen_location_ids)

        return problem_description, routes

    def __getitem__(self, idx: int) -> Tuple[ProblemDescription, Routes]:
        num_vehicles = self.num_vehicles if self.num_vehicles is not None else sps.randint(
            self.min_vehicles, self.max_vehicles + 1).rvs()
        random_problem_description, routes = self._get_random_problem_description_and_routes(num_vehicles)

        if self.transform is not None:
            random_problem_description, routes = self.transform(random_problem_description, routes)

        return random_problem_description, routes
