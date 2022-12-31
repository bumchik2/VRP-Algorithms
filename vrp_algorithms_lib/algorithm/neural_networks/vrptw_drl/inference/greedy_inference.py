import torch

from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.inference.sequential_inference import SequentialInference


class GreedyInference(SequentialInference):
    def choose_courier_idx(
            self,
            courier_logits: torch.Tensor
    ) -> int:
        return torch.argmax(courier_logits).item()

    def choose_location_idx(
            self,
            locations_logits: torch.Tensor
    ) -> int:
        return torch.argmax(locations_logits).item()
