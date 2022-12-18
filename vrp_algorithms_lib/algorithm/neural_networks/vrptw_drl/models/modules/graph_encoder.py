import torch
import torch.nn as nn


class GraphEncoder(nn.Module):
    def __init__(
            self,
            locations_information_dim: int,
            graph_encoder_hidden_dim: int,  # 128 in the original article
            graph_embedding_dim: int,
            num_heads: int,
            dropout: float
    ):
        """
        Encodes graph (information about locations positions and time windows)
        :param locations_information_dim: dimension for each location
        :param graph_encoder_hidden_dim: hidden dimension
        :param graph_embedding_dim: embedding dimension of MultiHeadAttention
        :param num_heads: number of heads in MultiHeadAttention
        :param dropout: dropout in MultiHeadAttention
        """
        super().__init__()

        self.fc = nn.Linear(locations_information_dim, graph_encoder_hidden_dim)

        self.fc_query = nn.Linear(graph_encoder_hidden_dim, graph_embedding_dim)
        self.fc_key = nn.Linear(graph_encoder_hidden_dim, graph_embedding_dim)
        self.fc_value = nn.Linear(graph_encoder_hidden_dim, graph_embedding_dim)

        self.multi_head_attention = nn.MultiheadAttention(
            graph_embedding_dim, num_heads, dropout=dropout
        )

        self.activation_function = nn.LeakyReLU()
        self.normalize_layer = nn.LayerNorm(graph_embedding_dim)

    def forward(
            self,
            locations_information: torch.Tensor
    ):
        # Input size: number_of_locations x locations_information_dim
        hidden = self.activation_function(self.fc(locations_information))
        # number_of_locations x graph_encoder_hidden_dim

        query = self.activation_function(self.fc_query(hidden))  # number_of_locations x graph_embedding_dim
        key = self.activation_function(self.fc_key(hidden))  # number_of_locations x graph_embedding_dim
        value = self.activation_function(self.fc_value(hidden))  # number_of_locations x graph_embedding_dim

        graph_embedding = self.multi_head_attention(query, key, value)[0]
        graph_embedding = self.normalize_layer(graph_embedding)

        return graph_embedding  # number_of_locations x graph_embedding_dim
