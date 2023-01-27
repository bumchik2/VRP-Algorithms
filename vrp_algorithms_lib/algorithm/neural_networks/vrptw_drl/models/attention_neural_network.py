"""
Article https://arxiv.org/pdf/2110.02629.pdf was taken as the basis of the model.
Note, that in the article Heterogeneous Capacitated VRP was solved.
However, the model below is quite flexible and may be used for multiple variations of the problem,
for example, for VRP with time windows.
"""

from typing import List

import torch
import torch.nn as nn

from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.models.modules.graph_encoder import GraphEncoder
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.models.modules.location_selection_decoder import \
    LocationSelectionDecoder, LocationSelectionDecoderWithCompatibilityLayer
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.models.modules.routes_encoder import RoutesEncoder
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.models.modules.vehicle_selection_decoder import \
    VehicleSelectionDecoder
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.models.modules.vehicles_state_encoder import \
    VehiclesStateEncoder


class AttentionNeuralNetwork(nn.Module):
    def __init__(
            self,
            locations_information_dim: int,
            vehicles_state_information_dim: int,
            graph_embedding_dim: int,
            locations_embedding_dim: int,
            dropout: float,
            use_compatibility_layer: bool,  # True in the original article
            graph_encoder_hidden_dim: int = 128,  # 128 in the original article
            vehicles_state_embedding_dim: int = 512,  # 512 in the original article
            routes_embedding_dim: int = 512,  # 512 in the original article
            num_heads: int = 8,  # 8 in the original article
            linear_blocks_number: int = 1,  # number of linear blocks in the end of each model.
            use_old_version=False,
    ):
        super().__init__()

        self.use_compatibility_layer = use_compatibility_layer

        self.graph_encoder = GraphEncoder(
            locations_information_dim=locations_information_dim,
            graph_encoder_hidden_dim=graph_encoder_hidden_dim,
            graph_embedding_dim=graph_embedding_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.vehicles_state_encoder = VehiclesStateEncoder(
            vehicles_state_information_dim=vehicles_state_information_dim,
            vehicles_state_embedding_dim=vehicles_state_embedding_dim,
            linear_blocks_number=linear_blocks_number,
            use_old_version=use_old_version
        )

        self.routes_encoder = RoutesEncoder(
            graph_embedding_dim=graph_embedding_dim,
            routes_embedding_dim=routes_embedding_dim,
            linear_blocks_number=linear_blocks_number,
            use_old_version=use_old_version
        )

        self.vehicle_selection_decoder = VehicleSelectionDecoder(
            vehicles_state_embedding_dim=vehicles_state_embedding_dim,
            routes_embedding_dim=routes_embedding_dim
        )

        if use_compatibility_layer:
            location_selection_decoder_class = LocationSelectionDecoderWithCompatibilityLayer
            linear_blocks_number = 1  # >1 not supported
        else:
            location_selection_decoder_class = LocationSelectionDecoder

        vehicle_state_information_dim_in_location_selection_decoder = vehicles_state_information_dim \
            if use_old_version else vehicles_state_embedding_dim
        self.location_selection_decoder = location_selection_decoder_class(
            graph_embedding_dim=graph_embedding_dim,
            vehicles_state_information_dim=vehicle_state_information_dim_in_location_selection_decoder,
            locations_embedding_dim=locations_embedding_dim,
            routes_embedding_dim=routes_embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            linear_blocks_number=linear_blocks_number
        )

    def get_graph_embedding(
            self,
            locations_information: torch.Tensor,
    ):
        # locations_information is of size number_of_locations x locations_information_dim
        graph_embedding = self.graph_encoder(locations_information=locations_information)
        return graph_embedding

    def get_vehicles_state_embedding(
            self,
            vehicles_state_information: torch.Tensor
    ):
        vehicles_state_embedding = self.vehicles_state_encoder(
            vehicles_state_information=vehicles_state_information
        )
        return vehicles_state_embedding

    def get_routes_embedding(
            self,
            graph_embedding: torch.Tensor,
            locations_idx: List[List[int]]
    ):
        routes_embedding = self.routes_encoder(
            graph_embedding=graph_embedding, locations_idx=locations_idx
        )
        return routes_embedding

    def get_vehicles_logits(
            self,
            vehicles_state_embedding: torch.Tensor,
            routes_embedding: torch.Tensor
    ):
        vehicles_logits = self.vehicle_selection_decoder(
            vehicles_state_embedding=vehicles_state_embedding,
            routes_embedding=routes_embedding
        )
        return vehicles_logits

    def get_locations_logits(
            self,
            graph_embedding: torch.Tensor,
            vehicles_state_information: torch.Tensor,
            routes_embedding: torch.Tensor,
            chosen_vehicle_idx: int,
            last_node_for_the_chosen_vehicle_idx: int
    ):
        return self.location_selection_decoder(
            graph_embedding=graph_embedding,
            vehicles_state_information=vehicles_state_information,
            routes_embedding=routes_embedding,
            chosen_vehicle_idx=chosen_vehicle_idx,
            last_node_for_the_chosen_vehicle_idx=last_node_for_the_chosen_vehicle_idx
        )
