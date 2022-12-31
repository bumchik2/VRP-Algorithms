import numpy as np
import torch

from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.inference.sequential_inference import SequentialInference


class SampleInference(SequentialInference):
    def choose_courier_idx(
            self,
            courier_logits: torch.Tensor
    ) -> int:
        p = torch.nn.Softmax()(courier_logits).to(torch.device('cpu')).numpy()
        return np.random.choice(range(len(courier_logits)), p=p)

    def choose_location_idx(
            self,
            locations_logits: torch.Tensor
    ) -> int:
        p = torch.nn.Softmax()(locations_logits).to(torch.device('cpu')).numpy()
        return np.random.choice(range(len(locations_logits)), p=p)
