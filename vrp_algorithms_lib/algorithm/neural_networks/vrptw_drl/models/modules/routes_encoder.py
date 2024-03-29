from typing import List

import torch
import torch.nn as nn

from vrp_algorithms_lib.algorithm.neural_networks.common.common_modules import LinearBlockWithNormalizationChain, \
    LinearBlockChain


class RoutesEncoder(nn.Module):
    def __init__(
            self,
            graph_embedding_dim: int,
            routes_embedding_dim: int,  # 512 in the original article
            linear_blocks_number: int,
            use_old_version=False
    ):
        """
        Gets embedding of the partial routes
        :param graph_embedding_dim: per-location dim in graph_embedding
        :param routes_embedding_dim: hidden_dimension
        """
        super().__init__()

        linear_block_chain_class = LinearBlockWithNormalizationChain if use_old_version else LinearBlockChain

        self.linear_blocks = linear_block_chain_class(
            input_dim=graph_embedding_dim,
            output_dim=routes_embedding_dim,
            linear_blocks_number=linear_blocks_number
        )

    def forward(
            self,
            graph_embedding: torch.Tensor,
            locations_idx: List[List[int]],
    ):
        # graph_embedding is of dim number_of_locations x graph_embedding_dim
        # locations_idx[i][j] shows the location idx added for vehicle i at step j
        # locations_idx size is number_of_vehicles x T,
        # where T is the current moment (number of locations distributed so far)

        route_embeddings: List[torch.Tensor] = []
        for locations_idx_for_route in locations_idx:
            route_embedding = graph_embedding[locations_idx_for_route].unsqueeze(0)
            route_embeddings.append(route_embedding)

        routes_embedding: torch.Tensor = torch.cat(route_embeddings, dim=0)  # vehicles_number x T x graph_embedding_dim
        routes_embedding = torch.max(routes_embedding, dim=1)[0]  # vehicles_number x graph_embedding_dim

        routes_embedding = self.linear_blocks(routes_embedding)  # vehicles_number x routes_embedding_dim
        return routes_embedding
