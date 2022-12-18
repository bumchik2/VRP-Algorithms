import torch.nn as nn
import torch


class LinearBlockWithNormalization(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int
    ):
        super().__init__()

        # input size is X x input_dim
        # output size is X x output_dim
        self.linear_block = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.LeakyReLU(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(
            self,
            x: torch.Tensor
    ):
        return self.linear_block(x)
