import torch.nn as nn
import torch


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

    def forward(
            self,
            locations_information: torch.Tensor
    ):
        # Input size: number_of_locations x locations_information_dim
        hidden = nn.ReLU()(self.fc(locations_information))  # number_of_locations x graph_encoder_hidden_dim

        query = nn.ReLU()(self.fc_query(hidden))  # number_of_locations x graph_embedding_dim
        key = nn.ReLU()(self.fc_key(hidden))  # number_of_locations x graph_embedding_dim
        value = nn.ReLU()(self.fc_value(hidden))  # number_of_locations x graph_embedding_dim

        graph_embedding = self.multi_head_attention(query, key, value, need_weights=False)
        return graph_embedding  # number_of_locations x graph_embedding_dim
