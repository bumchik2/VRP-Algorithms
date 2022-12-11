import torch.nn as nn
import torch


class VehiclesStateEncoder(nn.Module):
    def __init__(
            self,
            vehicles_state_information_dim: int,
            vehicles_state_embedding_dim: int,  # 512 in the original article
    ):
        """
        Encodes graph (information about locations positions and time windows)
        :param vehicles_state_information_dim: dimension for each vehicle
        :param vehicles_state_embedding_dim: hidden dimension
        """

        super().__init__()

        self.fc = nn.Linear(vehicles_state_information_dim, vehicles_state_embedding_dim)

    def forward(
            self,
            vehicles_state_information: torch.Tensor
    ):
        # Input size: vehicles_number x vehicles_state_information_dim
        hidden = nn.ReLU()(self.fc(vehicles_state_information))  # vehicles_number x vehicles_state_embedding_dim
        return hidden
