import os
import time
from collections import defaultdict
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch.optim
from IPython.display import clear_output
from tqdm import tqdm

import vrp_algorithms_lib.analytical_tools.viz_problem_description as my_viz
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.inference.greedy_inference import GreedyInference
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.models.model_base import ModelBase
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.objects import ProblemState, Action, \
    initialize_problem_state
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.train.dataset_with_prepared_solutions import \
    DatasetWithPreparedSolutions
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.train.simple_dataset import SimpleDataset
from vrp_algorithms_lib.problem.models import ProblemDescription
from vrp_algorithms_lib.problem.models import Routes
from vrp_algorithms_lib.problem.penalties.total_penalty_calculator import ALL_PENALTY_CALCULATORS


class TrainMode(Enum):
    PENALTY_MODE = 'penalty_mode'
    RECONSTRUCTION_MODE = 'reconstruction_mode'


def plot_learning_curves(history):
    plt.figure(figsize=(20, 6))

    plt.subplot(1, 4, 1)
    plt.title('Loss per epoch', fontsize=15)
    mean_epoch_loss_history = history['train']['mean_loss']
    plt.plot(range(1, len(mean_epoch_loss_history) + 1), mean_epoch_loss_history, label='mean')
    plt.plot(range(1, len(mean_epoch_loss_history) + 1), history['train']['5th_percentile_loss'],
             label='5th percentile')
    plt.plot(range(1, len(mean_epoch_loss_history) + 1), history['train']['95th_percentile_loss'],
             label='95th percentile')
    plt.xlabel('Epoch', fontsize=15)
    plt.legend()
    plt.grid(visible=True)

    plt.subplot(1, 4, 2)
    plt.title('Delta reward percentage per epoch, %', fontsize=15)
    mean_delta_reward_percentage_history = history['train']['mean_delta_reward_percentage']
    plt.plot(range(1, len(mean_delta_reward_percentage_history) + 1), mean_delta_reward_percentage_history,
             label='mean')
    plt.plot(range(1, len(mean_delta_reward_percentage_history) + 1), history['train'][
        '5th_percentile_delta_reward_percentage'], label='5th percentile')
    plt.plot(range(1, len(mean_delta_reward_percentage_history) + 1), history['train'][
        '95th_percentile_delta_reward_percentage'], label='95th percentile')
    plt.xlabel('Epoch', fontsize=15)
    plt.legend()
    plt.grid(visible=True)

    plt.subplot(1, 4, 3)
    plt.title('Problem loss', fontsize=15)
    problem_loss_history = history['train']['mean_problem_loss']
    plt.plot(range(1, len(problem_loss_history) + 1), problem_loss_history)
    plt.grid(visible=True)

    plt.subplot(1, 4, 4)
    plt.title('Mean incorrect locations choices share', fontsize=15)
    incorrect_location_choices = history['train']['mean_incorrect_location_choices_share']
    plt.plot(range(1, len(incorrect_location_choices) + 1), incorrect_location_choices)
    plt.grid(visible=True)


def train_one_problem(
        model: Union[ModelBase, torch.nn.Module],
        trainer: ModelBase,
        criterion,
        optimizer: torch.optim.Optimizer,
        problem_description: ProblemDescription,
        routes: Optional[Routes],
        device: torch.device,
        train_mode: TrainMode
) -> Dict[str, Any]:
    model.train()

    optimizer.zero_grad()

    problem_state: ProblemState = initialize_problem_state(problem_description=problem_description)

    model.initialize(problem_state, None)
    trainer.initialize(problem_state, routes)

    total_loss = torch.tensor(0.).to(device)
    incorrect_location_choices = 0
    total_delta_reward = 0
    total_trainer_reward = 0

    for step_number in range(len(problem_description.locations)):
        model_couriers_logits = model.get_couriers_logits(problem_state=problem_state)
        trainer_couriers_logits = trainer.get_couriers_logits(problem_state=problem_state)

        model_courier_idx = torch.argmax(model_couriers_logits).item()
        model_courier_id = problem_state.idx_to_courier_id[model_courier_idx]
        trainer_courier_idx = torch.argmax(trainer_couriers_logits).item()
        trainer_courier_id = problem_state.idx_to_courier_id[trainer_courier_idx]
        chosen_courier_id = trainer_courier_id

        model_locations_logits = model.get_locations_logits(
            courier_id=chosen_courier_id, problem_state=problem_state
        )
        trainer_locations_logits = trainer.get_locations_logits(
            courier_id=chosen_courier_id, problem_state=problem_state
        )

        model_location_idx = torch.argmax(model_locations_logits).item()
        model_location_id = problem_state.idx_to_location_id[model_location_idx]
        trainer_location_idx = torch.argmax(trainer_locations_logits).item()
        trainer_location_id = problem_state.idx_to_location_id[trainer_location_idx]

        if model_location_id in problem_state.visited_location_ids:
            incorrect_location_choices += 1

        model_action = Action(courier_id=model_courier_id, location_id=model_location_id)
        trainer_action = Action(courier_id=trainer_courier_id, location_id=trainer_location_id)

        model_reward = problem_state.get_reward(action=model_action, routes=routes)
        trainer_reward = problem_state.get_reward(action=trainer_action, routes=routes)
        courier_choice_loss = criterion(model_couriers_logits, torch.tensor(trainer_courier_idx).to(device))
        location_choice_loss = criterion(model_locations_logits, torch.tensor(trainer_location_idx).to(device))

        if train_mode == TrainMode.PENALTY_MODE:
            delta_reward = (trainer_reward - model_reward)
        elif train_mode == TrainMode.RECONSTRUCTION_MODE:
            delta_reward = 10
        else:
            raise ValueError(f'Unknown TrainMode: {train_mode}')

        total_delta_reward += delta_reward
        total_trainer_reward += trainer_reward
        total_loss += (courier_choice_loss + location_choice_loss) * delta_reward

        problem_state.update(trainer_action)

    total_loss /= len(problem_description.locations)
    total_loss.backward()
    optimizer.step()

    average_problem_loss = total_loss.item()
    delta_reward_percentage = 100.0 * total_delta_reward / np.abs(total_trainer_reward)

    return {
        'incorrect_location_choices_share': incorrect_location_choices / len(problem_description.locations),
        'mean_problem_loss': average_problem_loss,
        'delta_reward_percentage': delta_reward_percentage
    }


@torch.no_grad()
def get_and_plot_inference_examples(
        model: Union[ModelBase, torch.nn.Module],
        problem_description_samples: List[List[Tuple[ProblemDescription, Routes]]],
        suptitle: Optional[str] = None
):
    if isinstance(model, torch.nn.Module):
        model.eval()

    fig = plt.figure(figsize=(20, 7 * len(problem_description_samples)))
    if suptitle:
        fig.suptitle(suptitle, fontsize=22)

    for batch_number, problem_description_batch in enumerate(problem_description_samples):
        for i, (problem_description, routes) in enumerate(problem_description_batch):
            greedy_inference = GreedyInference(
                model=model,
                problem_description=problem_description,
                routes=routes
            )
            routes = greedy_inference.solve_problem()

            plt.subplot(len(problem_description_samples), 3, 3 * batch_number + i + 1)
            penalties = {
                penalty.get_penalty_name(): penalty.calculate(problem_description, routes)
                for penalty in ALL_PENALTY_CALCULATORS
            }

            losses_str = '\n'.join(
                [f'{penalty_name}: {round(penalties[penalty_name], 2)}' for penalty_name in penalties])
            title = f'Inference example {i + 1}\n' + losses_str
            my_viz.plot_routes(problem_description, routes, ax=plt.gca(), title=title, legend=False)

    plt.tight_layout()


def train(
        model: Union[ModelBase, torch.nn.Module],
        trainer: ModelBase,
        criterion,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        dataset: Union[SimpleDataset, DatasetWithPreparedSolutions],
        problem_description_samples: List[List[Tuple[ProblemDescription, Routes]]],
        device: torch.device,
        train_mode: TrainMode,
        checkpoint_path: Optional[Union[os.PathLike, str]] = None,
        lr_scheduler=None,
):
    history = defaultdict(lambda: defaultdict(list))

    for epoch in tqdm(range(num_epochs), position=0, leave=True):
        epoch_start_time = time.time()

        problem_losses = []
        delta_reward_percentages = []
        incorrect_locations_choices_share = []

        for i, (problem_description, routes) in tqdm(enumerate(dataset), total=len(dataset), position=0, leave=False):
            train_epoch_info = train_one_problem(
                model=model,
                trainer=trainer,
                criterion=criterion,
                optimizer=optimizer,
                problem_description=problem_description,
                routes=routes,
                device=device,
                train_mode=train_mode
            )
            for key in ['mean_problem_loss']:
                history['train'][key].append(train_epoch_info[key])

            incorrect_locations_choices_share.append(train_epoch_info['incorrect_location_choices_share'])
            problem_losses.append(train_epoch_info['mean_problem_loss'])
            delta_reward_percentages.append(train_epoch_info['delta_reward_percentage'])

            if i + 1 == len(dataset):
                break

        if lr_scheduler is not None:
            lr_scheduler.step()

        if checkpoint_path:
            torch.save(model.state_dict(), checkpoint_path)

        history['train']['mean_incorrect_location_choices_share'].append(np.mean(incorrect_locations_choices_share))

        history['train']['mean_loss'].append(np.mean(problem_losses))
        history['train']['5th_percentile_loss'].append(np.quantile(problem_losses, 0.05))
        history['train']['95th_percentile_loss'].append(np.quantile(problem_losses, 0.95))

        history['train']['mean_delta_reward_percentage'].append(np.mean(delta_reward_percentages))
        history['train']['5th_percentile_delta_reward_percentage'].append(np.quantile(delta_reward_percentages, 0.05))
        history['train']['95th_percentile_delta_reward_percentage'].append(np.quantile(delta_reward_percentages, 0.95))

        clear_output()
        print(f'Epoch {epoch + 1} of {num_epochs} took {round(time.time() - epoch_start_time, 3)} seconds')
        plot_learning_curves(history)
        get_and_plot_inference_examples(trainer, problem_description_samples, suptitle='Trainer inference example')
        get_and_plot_inference_examples(model, problem_description_samples, suptitle='Model inference example')
        plt.show()

    return history
