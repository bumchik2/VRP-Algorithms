from typing import Optional
from typing import Tuple

import scipy.stats as sps
import torch.utils.data as data_utils

import vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.train.common_utils as common_utils
from vrp_algorithms_lib.problem.models import ProblemDescription


class SimpleDataset(data_utils.Dataset):
    def __init__(
            self,
            num_vehicles: Optional[int] = None,
            min_vehicles: Optional[int] = None,
            max_vehicles: Optional[int] = None,
            num_locations: Optional[int] = None,
            min_locations: Optional[int] = None,
            max_locations: Optional[int] = None,
            dataset_size: int = 100
    ):
        super().__init__()

        common_utils.common_min_max_value_check(num_vehicles, min_vehicles, max_vehicles)
        common_utils.common_min_max_value_check(num_locations, min_locations, max_locations)

        self.num_vehicles = num_vehicles
        self.min_vehicles = min_vehicles
        self.max_vehicles = max_vehicles

        self.num_locations = num_locations
        self.min_locations = min_locations
        self.max_locations = max_locations

        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx: int) -> Tuple[ProblemDescription, None]:
        num_vehicles = self.num_vehicles if self.num_vehicles is not None else sps.randint(
            self.min_vehicles, self.max_vehicles + 1).rvs()
        num_locations = self.num_locations if self.num_locations is not None else sps.randint(
            self.min_locations, self.max_locations + 1).rvs()

        random_problem_description = common_utils.get_random_problem_description(
            num_vehicles=num_vehicles,
            num_locations=num_locations
        )
        return random_problem_description, None
