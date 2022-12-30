import torch
import torch.nn as nn

from vrp_algorithms_lib.algorithm.neural_networks.common.common_modules import LinearBlockWithNormalizationChain


class VehiclesStateEncoder(nn.Module):
    def __init__(
            self,
            vehicles_state_information_dim: int,
            vehicles_state_embedding_dim: int,  # 512 in the original article
            linear_blocks_number: int
    ):
        """
        Encodes graph (information about locations positions and time windows)
        :param vehicles_state_information_dim: dimension for each vehicle
        :param vehicles_state_embedding_dim: hidden dimension
        """
        super().__init__()

        self.linear_blocks = LinearBlockWithNormalizationChain(
            input_dim=vehicles_state_information_dim,
            output_dim=vehicles_state_embedding_dim,
            linear_blocks_number=linear_blocks_number
        )

    def forward(
            self,
            vehicles_state_information: torch.Tensor
    ):
        # Input size: vehicles_number x vehicles_state_information_dim
        hidden = self.linear_blocks(vehicles_state_information)  # vehicles_number x vehicles_state_embedding_dim
        return hidden
