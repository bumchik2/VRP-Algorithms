import torch
import torch.nn as nn


class LinearBlock(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int
    ):
        super().__init__()

        # input size is X x input_dim
        # output size is X x output_dim
        self.linear_block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU()
        )

    def forward(
            self,
            x: torch.Tensor
    ):
        return self.linear_block(x)


class LinearBlockChain(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            linear_blocks_number: int
    ):
        # input size is X x input_dim
        # output size is X x output_dim
        super().__init__()

        linear_blocks = []
        for i in range(linear_blocks_number - 1):
            linear_blocks.append(LinearBlock(
                input_dim * 2 ** i, input_dim * 2 ** (i + 1)))
        linear_blocks.append(LinearBlock(
            input_dim * 2 ** (linear_blocks_number - 1), output_dim))

        self.linear_blocks = nn.Sequential(*linear_blocks)

    def forward(
            self,
            x: torch.Tensor
    ):
        return self.linear_blocks(x)


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


class LinearBlockWithNormalizationChain(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            linear_blocks_number: int
    ):
        # input size is X x input_dim
        # output size is X x output_dim
        super().__init__()

        linear_blocks = []
        for i in range(linear_blocks_number - 1):
            linear_blocks.append(LinearBlockWithNormalization(
                input_dim * 2 ** i, input_dim * 2 ** (i + 1)))
        linear_blocks.append(LinearBlockWithNormalization(
            input_dim * 2 ** (linear_blocks_number - 1), output_dim))

        self.linear_blocks = nn.Sequential(*linear_blocks)

    def forward(
            self,
            x: torch.Tensor
    ):
        return self.linear_blocks(x)
