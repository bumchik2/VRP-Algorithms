import torch
import torch.nn as nn

from vrp_algorithms_lib.algorithm.neural_networks.common.common_modules import LinearBlockWithNormalization, \
    LinearBlockWithNormalizationChain


class LocationSelectionDecoderWithCompatibilityLayer(nn.Module):
    def __init__(
            self,
            graph_embedding_dim: int,
            vehicles_state_information_dim: int,
            locations_embedding_dim: int,
            routes_embedding_dim: int,
            num_heads: int,
            dropout: float,
            linear_blocks_number: int
    ):
        super().__init__()

        assert linear_blocks_number == 1  # no extra linear blocks are supported!

        self.fc_query = nn.Linear(
            2 * graph_embedding_dim + routes_embedding_dim + vehicles_state_information_dim,
            locations_embedding_dim
        )
        self.fc_key = nn.Linear(graph_embedding_dim, locations_embedding_dim)
        self.fc_value = nn.Linear(graph_embedding_dim, locations_embedding_dim)

        self.fc_compatibility = LinearBlockWithNormalization(graph_embedding_dim, locations_embedding_dim)

        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=locations_embedding_dim,
            num_heads=num_heads,
            dropout=dropout
        )

    def forward(
            self,
            graph_embedding: torch.Tensor,
            vehicles_state_information: torch.Tensor,
            routes_embedding: torch.Tensor,
            chosen_vehicle_idx: int,
            last_node_for_the_chosen_vehicle_idx: int
    ):
        number_of_locations = graph_embedding.size()[0]
        graph_embedding_mean = torch.mean(graph_embedding, dim=0)  # graph_embedding_dim
        last_node_embedding = graph_embedding[last_node_for_the_chosen_vehicle_idx]  # graph_embedding_dim
        route_embedding = routes_embedding[chosen_vehicle_idx]  # routes_embedding_dim
        vehicle_state_information = vehicles_state_information[chosen_vehicle_idx]  # vehicles_state_information_dim

        graph_embedding_expanded = torch.cat(
            [graph_embedding_mean, last_node_embedding, route_embedding, vehicle_state_information], dim=0
        )  # graph_embedding_dim * 2 + routes_embedding + vehicles_state_information_dim
        graph_embedding_expanded = graph_embedding_expanded.repeat(number_of_locations, 1)

        query = self.fc_query(graph_embedding_expanded)
        key = self.fc_key(graph_embedding)
        value = self.fc_value(graph_embedding)

        locations_embedding = self.multi_head_attention(query, key, value)[0]
        # number_of_locations x locations_embedding_dim
        locations_embedding = locations_embedding.unsqueeze(1)

        compatibility_embeddings = self.fc_compatibility(graph_embedding)
        compatibility_embeddings = compatibility_embeddings.unsqueeze(2)

        logits = 10 * torch.tanh(
            torch.bmm(locations_embedding, compatibility_embeddings).squeeze()
        )[:-1]  # number_of_locations
        # remove the last logit for depot
        return logits


class LocationSelectionDecoder(nn.Module):
    def __init__(
            self,
            graph_embedding_dim: int,
            vehicles_state_information_dim: int,
            locations_embedding_dim: int,
            routes_embedding_dim: int,
            num_heads: int,
            dropout: float,
            linear_blocks_number: int
    ):
        super().__init__()

        self.fc_query = nn.Linear(
            2 * graph_embedding_dim + routes_embedding_dim + vehicles_state_information_dim,
            locations_embedding_dim
        )
        self.fc_key = nn.Linear(graph_embedding_dim, locations_embedding_dim)
        self.fc_value = nn.Linear(graph_embedding_dim, locations_embedding_dim)

        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=locations_embedding_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        if linear_blocks_number > 1:
            self.extra_linear_blocks = LinearBlockWithNormalizationChain(
                input_dim=locations_embedding_dim,
                output_dim=locations_embedding_dim,
                linear_blocks_number=linear_blocks_number - 1
            )
        else:
            self.extra_linear_blocks = None

        self.fc = nn.Linear(locations_embedding_dim, 1)

    def forward(
            self,
            graph_embedding: torch.Tensor,
            vehicles_state_information: torch.Tensor,
            routes_embedding: torch.Tensor,
            chosen_vehicle_idx: int,
            last_node_for_the_chosen_vehicle_idx: int
    ):
        # graph embedding is of size number_of_locations x graph_embedding_dim
        # vehicles_state_information is of size number_of_vehicles x vehicles_state_information_dim
        number_of_locations = graph_embedding.size()[0]
        vehicle_state_information = vehicles_state_information[chosen_vehicle_idx]. \
            unsqueeze(0).repeat(number_of_locations, 1)

        last_node_embedding = graph_embedding[last_node_for_the_chosen_vehicle_idx]. \
            unsqueeze(0).repeat(number_of_locations, 1)

        # route_embedding was not passed into MHA in the original article.
        # However, we need to pass it, if we want to consider route geometry when choosing the next node
        route_embedding = routes_embedding[chosen_vehicle_idx]. \
            unsqueeze(0).repeat(number_of_locations, 1)

        # graph embedding with information about the chosen vehicle
        # graph_embedding_expanded is of size { number_of_locations x
        # (2 * graph_embedding_dim + vehicles_state_information_dim + routes_embedding_dim) }
        graph_embedding_expanded = torch.cat(
            [graph_embedding, vehicle_state_information, last_node_embedding, route_embedding], dim=1
        )

        query = self.fc_query(graph_embedding_expanded)
        key = self.fc_key(graph_embedding)
        value = self.fc_value(graph_embedding)

        locations_embedding = self.multi_head_attention(query, key, value)[0]
        # number_of_locations x locations_embedding_dim

        if self.extra_linear_blocks is not None:
            locations_embedding = self.extra_linear_blocks(locations_embedding)

        logits = self.fc(locations_embedding)[:-1, 0]  # remove the last logit for depot
        return logits
