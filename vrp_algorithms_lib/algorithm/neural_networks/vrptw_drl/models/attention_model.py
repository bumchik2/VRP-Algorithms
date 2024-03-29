from typing import List
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.models.attention_neural_network import \
    AttentionNeuralNetwork
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.models.model_base import ModelBase
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.objects import CourierId
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.objects import ProblemState
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.objects import Routes


class AttentionModel(nn.Module, ModelBase):
    def __init__(
            self,
            attention_neural_network: AttentionNeuralNetwork,
            use_max_time_window_start_s: bool,
            device: torch.device = torch.device('cpu'),
            secure_mode: bool = True,
            use_old_version=False
    ):
        super().__init__()

        self.device = device
        self.attention_neural_network = attention_neural_network

        self.vehicles_state_embedding = None
        self.vehicles_state_information = None
        self.routes_embedding = None
        self.locations_mean_lat = None
        self.locations_mean_lon = None

        self.use_max_time_window_start_s = use_max_time_window_start_s
        self.secure_mode = secure_mode
        self.use_old_version = use_old_version

        self.to(device)

    def normalize_lat(self, lat: float):
        return (lat - self.locations_mean_lat) * 6.0

    def normalize_lon(self, lon: float):
        return (lon - self.locations_mean_lon) * 6.0

    def get_locations_information(
            self,
            problem_state: ProblemState
    ) -> torch.Tensor:
        # return torch.Tensor of size number_of_locations x locations_information_dim
        # the last location is depot, it is marked with a special flag
        locations_information = []

        for location_idx in range(len(problem_state.problem_description.locations)):
            location_id = problem_state.idx_to_location_id[location_idx]
            location = problem_state.problem_description.locations[location_id]

            time_window_start_s_normalized = location.time_window_start_s / 86400.0
            time_window_end_s_normalized = location.time_window_end_s / 86400.0
            normalized_lat = self.normalize_lat(location.point.lat)
            normalized_lon = self.normalize_lon(location.point.lon)
            location_requires_visit = (location_id not in problem_state.visited_location_ids)

            location_information = [
                normalized_lat,
                normalized_lon,
                time_window_start_s_normalized,
                time_window_end_s_normalized,
                location_requires_visit
            ]
            locations_information.append(location_information)

        depot = list(problem_state.problem_description.depots.values())[0]
        depot_normalized_lat = self.normalize_lat(depot.point.lat)
        depot_normalized_lon = self.normalize_lon(depot.point.lon)
        depot_information = [
            depot_normalized_lat,
            depot_normalized_lon,
            0.0,  # normalized time window start
            1.0,  # normalized time window end
            0.0  # no demand for depot
        ]
        locations_information.append(depot_information)

        return torch.tensor(locations_information, dtype=torch.float32).to(self.device)  # (n + 1) x dim_1

    def get_vehicles_state_information(
            self,
            problem_state: ProblemState
    ) -> torch.Tensor:
        # return torch.Tensor of size number_of_vehicles x vehicles_state_information_dim
        vehicles_state_information = []

        for courier_idx in range(len(problem_state.problem_description.couriers)):
            vehicle_state = problem_state.vehicle_states[courier_idx]

            last_location_idx_in_route = problem_state.locations_idx[courier_idx][-1]
            if last_location_idx_in_route == len(problem_state.problem_description.locations):  # depot idx
                depot = list(problem_state.problem_description.depots.values())[0]
                last_point_lat = depot.point.lat
                last_point_lon = depot.point.lon
            else:
                last_location_id_in_route = problem_state.idx_to_location_id[last_location_idx_in_route]
                last_location_in_route = problem_state.problem_description.locations[last_location_id_in_route]
                last_point_lat = last_location_in_route.point.lat
                last_point_lon = last_location_in_route.point.lon

            normalized_last_point_lat = self.normalize_lat(last_point_lat)
            normalized_last_point_lon = self.normalize_lon(last_point_lon)
            # divide distance in kilometers by 100
            normalized_total_travel_distance = vehicle_state.total_distance / 100.0
            # divide number of orders delivered by some heuristic constant
            normalized_number_of_orders = (len(set(vehicle_state.partial_route)) - 1) / 50.0

            vehicle_information = [
                normalized_last_point_lat,
                normalized_last_point_lon,
                normalized_total_travel_distance,
                normalized_number_of_orders
            ]

            if self.use_max_time_window_start_s:
                normalized_max_time_window_start_s = 0.
                for location_id in vehicle_state.get_filtered_partial_route():
                    location = problem_state.problem_description.locations[location_id]
                    normalized_max_time_window_start_s = max(normalized_max_time_window_start_s,
                                                             location.time_window_start_s)
                normalized_max_time_window_start_s /= 86400.0
                vehicle_information.append(normalized_max_time_window_start_s)

            vehicles_state_information.append(vehicle_information)

        return torch.tensor(vehicles_state_information, dtype=torch.float32).to(self.device)

    def get_graph_embedding(
            self,
            problem_state: ProblemState
    ):
        locations_information = self.get_locations_information(problem_state=problem_state)
        graph_embedding = self.attention_neural_network.get_graph_embedding(
            locations_information=locations_information
        )
        return graph_embedding

    def initialize(
            self,
            problem_state: ProblemState,
            routes: Optional[Routes]
    ):
        self.locations_mean_lat = np.mean(
            [location.point.lat for location in problem_state.problem_description.locations.values()])
        self.locations_mean_lon = np.mean(
            [location.point.lon for location in problem_state.problem_description.locations.values()])

    @staticmethod
    def get_locations_idx(
            problem_state: ProblemState
    ) -> List[List[int]]:
        return problem_state.locations_idx

    def _get_couriers_logits(self, problem_state: ProblemState) -> torch.tensor:
        self.vehicles_state_information = self.get_vehicles_state_information(problem_state=problem_state)

        self.vehicles_state_embedding = self.attention_neural_network.get_vehicles_state_embedding(
            vehicles_state_information=self.vehicles_state_information
        )

        locations_idx = AttentionModel.get_locations_idx(problem_state=problem_state)

        graph_embedding = self.get_graph_embedding(problem_state=problem_state)

        self.routes_embedding = self.attention_neural_network.get_routes_embedding(
            graph_embedding=graph_embedding,
            locations_idx=locations_idx
        )

        result = self.attention_neural_network.get_vehicles_logits(
            vehicles_state_embedding=self.vehicles_state_embedding,
            routes_embedding=self.routes_embedding
        )

        return result

    def _get_locations_logits(self, courier_id: CourierId, problem_state: ProblemState) -> torch.tensor:
        locations_idx = AttentionModel.get_locations_idx(problem_state=problem_state)

        courier_idx = problem_state.courier_id_to_idx[courier_id]

        graph_embedding = self.get_graph_embedding(problem_state=problem_state)

        vehicles_state_information_in_get_locations_logits = self.vehicles_state_information if self.use_old_version \
            else self.vehicles_state_embedding

        result = self.attention_neural_network.get_locations_logits(
            graph_embedding=graph_embedding,
            vehicles_state_information=vehicles_state_information_in_get_locations_logits,
            routes_embedding=self.routes_embedding,
            chosen_vehicle_idx=courier_idx,
            last_node_for_the_chosen_vehicle_idx=locations_idx[courier_idx][-1]
        )

        if self.secure_mode:
            self.vehicles_state_information = None
            self.routes_embedding = None
            self.vehicles_state_embedding = None

        return result
