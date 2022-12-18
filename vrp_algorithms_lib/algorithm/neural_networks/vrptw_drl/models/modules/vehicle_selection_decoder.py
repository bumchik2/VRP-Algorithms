import torch
import torch.nn as nn


class VehicleSelectionDecoder(nn.Module):
    def __init__(
            self,
            vehicles_state_embedding_dim: int,
            routes_embedding_dim: int
    ):
        super().__init__()
        self.fc = nn.Linear(vehicles_state_embedding_dim + routes_embedding_dim, 1)

    def forward(
            self,
            vehicles_state_embedding: torch.Tensor,
            routes_embedding: torch.Tensor
    ):
        # vehicles_state_embedding is of size vehicles_number x vehicles_state_information_dim
        # routes_embedding is of size vehicles_number x routes_embedding_dim

        assert vehicles_state_embedding.size()[0] == routes_embedding.size()[0]

        vehicles_routes_embedding = torch.cat(
            [vehicles_state_embedding, routes_embedding], dim=1
        )
        logits = self.fc(vehicles_routes_embedding)[:, 0]
        return logits
