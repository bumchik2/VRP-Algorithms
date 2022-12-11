"""
Article https://arxiv.org/pdf/2110.02629.pdf was taken as the basis of the model.
Note, that in the article Heterogeneous Capacitated VRP was solved.
However, the model below is quite flexible and may be used for multiple variations of the problem,
for example, for VRP with time windows.
"""

import torch
import torch.nn as nn
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.models.modules.graph_encoder import GraphEncoder
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.models.modules.routes_encoder import RoutesEncoder
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.models.modules.vehicles_state_encoder import \
    VehiclesStateEncoder
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.models.modules.vehicle_selection_decoder import \
    VehicleSelectionDecoder
from typing import List


class AttentionModel(nn.Module):
    def __init__(
            self,
            locations_information_dim: int,
            vehicles_state_information_dim: int,
            graph_embedding_dim: int,
            dropout: float,
            graph_encoder_hidden_dim: int = 128,  # 128 in the original article
            vehicles_state_hidden_dim: int = 512,  # 512 in the original article
            routes_embedding_dim: int = 512,  # 512 in the original article
            num_heads: int = 8,  # 8 in the original article
    ):
        super().__init__()

        self.graph_encoder = GraphEncoder(
            locations_information_dim=locations_information_dim,
            graph_encoder_hidden_dim=graph_encoder_hidden_dim,
            graph_embedding_dim=graph_embedding_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.vehicles_state_encoder = VehiclesStateEncoder(
            vehicles_state_information_dim=vehicles_state_information_dim,
            vehicles_state_hidden_dim=vehicles_state_hidden_dim
        )

        self.routes_encoder = RoutesEncoder(
            graph_embedding_dim=graph_embedding_dim,
            routes_embedding_dim=routes_embedding_dim
        )

        self.vehicle_selection_decoder = VehicleSelectionDecoder(
            vehicles_state_information_dim=vehicles_state_information_dim,
            routes_embedding_dim=routes_embedding_dim
        )

    def get_graph_embedding(
            self,
            locations_information: torch.Tensor,
    ):
        # locations_information is of size number_of_locations x locations_information_dim
        graph_embedding = self.graph_encoder(locations_information=locations_information)
        return graph_embedding

    def forward(
            self,
            graph_embedding: torch.Tensor,
            vehicles_state_information: torch.Tensor,
            locations_idx: List[List[int]]
    ):
        # graph embedding is of size number_of_locations x graph_embedding_dim
        vehicles_state_embedding = self.vehicles_state_encoder(
            vehicles_state_information=vehicles_state_information
        )
        routes_embedding = self.routes_encoder(
            graph_embedding=graph_embedding, locations_idx=locations_idx
        )
        vehicles_logits = self.vehicle_selection_decoder(
            vehicles_state_embedding=vehicles_state_embedding, routes_embedding=routes_embedding
        )
        # TODO: get locations_logits (возможно, нужно будет forward разбить на две части:
        #  сначала отдельно возвращаем вероятности для машины, затем - для локации)
