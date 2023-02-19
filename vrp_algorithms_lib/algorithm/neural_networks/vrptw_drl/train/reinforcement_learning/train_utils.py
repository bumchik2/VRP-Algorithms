import os
import time
from collections import defaultdict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch.optim
from tqdm import tqdm

from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.inference.sample_inference import SampleInference
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.models.model_base import ModelBase
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.train.dataset_with_prepared_solutions import \
    DatasetWithPreparedSolutions, DatasetWithFixedPreparedSolutions
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.train.simple_dataset import SimpleDataset
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.train.train_utils import train, TrainMode, \
    clear_output_and_draw_statistics
from vrp_algorithms_lib.problem.models import ProblemDescription
from vrp_algorithms_lib.problem.models import Routes
from vrp_algorithms_lib.problem.penalties.total_penalty_calculator import TotalPenaltyCalculator


def train_rl(
        model: Union[ModelBase, torch.nn.Module],
        trainer: ModelBase,
        criterion,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        dataset: Union[SimpleDataset, DatasetWithPreparedSolutions, DatasetWithFixedPreparedSolutions],
        problem_description_samples: List[List[Tuple[ProblemDescription, Routes]]],
        device: torch.device,
        number_of_trajectories_per_problem_description: int,
        percentile_best: float,
        checkpoint_path: Optional[Union[os.PathLike, str]] = None,
        lr_scheduler=None,
):
    history = defaultdict(lambda: defaultdict(list))
    total_penalty_calculator = TotalPenaltyCalculator()
    epoch_start_time = time.time()

    for _ in tqdm(range(num_epochs)):
        problem_description, _ = dataset[0]

        # Generate several trajectories
        routes_list = []
        for _ in range(number_of_trajectories_per_problem_description):
            inference = SampleInference(
                model=model,
                problem_description=problem_description,
                routes=None
            )
            routes = inference.solve_problem()
            routes_list.append(
                routes
            )

        # Calculate total penalties of the routes
        routes_metrics = [total_penalty_calculator.calculate(problem_description, routes) for routes in routes_list]

        # Select percentile_best best trajectories
        number_of_best_routes = int(percentile_best * number_of_trajectories_per_problem_description)
        best_routes_idx = np.argpartition(routes_metrics, number_of_best_routes)[:number_of_best_routes]
        best_routes_list = []
        for idx in best_routes_idx:
            best_routes_list.append(routes_list[idx])

        # Create dataset with the best trajectories
        dataset_part = DatasetWithFixedPreparedSolutions(
            [problem_description] * len(best_routes_list),
            best_routes_list
        )

        # Train on these best trajectories
        history = train(
            model=model,
            trainer=trainer,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=1,
            dataset=dataset_part,
            problem_description_samples=problem_description_samples,
            device=device,
            checkpoint_path=checkpoint_path,
            lr_scheduler=lr_scheduler,
            train_mode=TrainMode.RECONSTRUCTION_MODE,
            history=history,
            need_to_clear_output_and_draw_statistics=False
        )

        clear_output_and_draw_statistics(
            epoch_start_time=epoch_start_time,
            history=history,
            num_epochs=num_epochs,
            trainer=trainer,
            model=model,
            problem_description_samples=problem_description_samples
        )

    return history
