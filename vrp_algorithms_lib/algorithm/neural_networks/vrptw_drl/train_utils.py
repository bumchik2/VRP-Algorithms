import time
from collections import defaultdict
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps
import torch.optim
import torch.utils.data as data_utils
from IPython.display import clear_output
from tqdm import tqdm

from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.models.model_base import ModelBase
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.objects import ProblemState, Action, \
    initialize_problem_state
from vrp_algorithms_lib.problem.models import ProblemDescription, Courier, Location, Depot, DepotId, CourierId, \
    LocationId, Penalties
from vrp_algorithms_lib.problem.models import get_geodesic_time_matrix, get_euclidean_distance_matrix


def plot_learning_curves(history):
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.title('Loss per epoch', fontsize=15)
    mean_epoch_loss_history = history['train']['mean_epoch_loss']
    plt.plot(range(1, len(mean_epoch_loss_history) + 1), mean_epoch_loss_history, label='mean')
    plt.plot(range(1, len(mean_epoch_loss_history) + 1), history['train']['5th_percentile_epoch_loss'],
             label='5th percentile')
    plt.plot(range(1, len(mean_epoch_loss_history) + 1), history['train']['95th_percentile_epoch_loss'],
             label='95th percentile')
    plt.xlabel('Epoch', fontsize=15)
    plt.legend()
    plt.grid(visible=True)

    plt.subplot(1, 2, 2)
    plt.title('Problem loss', fontsize=15)
    problem_loss_history = history['train']['problem_loss']
    plt.plot(range(1, len(problem_loss_history) + 1), problem_loss_history)
    plt.grid(visible=True)
    plt.show()


def get_random_problem_description(
        num_vehicles: int,
        num_locations: int
) -> ProblemDescription:
    mean_lat = 55.752572
    mean_lon = 37.622269
    lat_distribution = sps.uniform(loc=mean_lat - 0.15, scale=0.3)
    lon_distribution = sps.uniform(loc=mean_lon - 0.15, scale=0.3)

    depots: Dict[DepotId, Depot] = {
        DepotId('depot 1'): Depot(
            id=DepotId('depot 1'),
            lat=lat_distribution.rvs(),
            lon=lon_distribution.rvs()
        )
    }

    locations: Dict[LocationId, Location] = {
        LocationId(f'location {i + 1}'): Location(
            id=LocationId(f'location {i + 1}'),
            depot_id=DepotId('depot 1'),
            lat=lat_distribution.rvs(),
            lon=lon_distribution.rvs(),
            time_window_start_s=0,
            time_window_end_s=0
        )
        for i in range(num_locations)
    }

    couriers: Dict[CourierId, Courier] = {
        CourierId(f'courier {i + 1}'): Courier(
            id=CourierId(f'courier {i + 1}')
        )
        for i in range(num_vehicles)
    }

    penalties = Penalties(
        distance_penalty_multiplier=8,
        global_proximity_factor=0,
        out_of_time_penalty_per_minute=0
    )

    distance_matrix = get_euclidean_distance_matrix(depots=depots, locations=locations)
    time_matrix = get_geodesic_time_matrix(depots=depots, locations=locations, distance_matrix=distance_matrix)

    problem_description = ProblemDescription(
        depots=depots,
        locations=locations,
        couriers=couriers,
        penalties=penalties,
        distance_matrix=distance_matrix,
        time_matrix=time_matrix
    )

    return problem_description


class SimpleDataset(data_utils.Dataset):
    def __init__(
            self,
            num_vehicles: int,
            num_locations: int,
            dataset_size: int = 100
    ):
        super().__init__()
        self.num_vehicles = num_vehicles
        self.num_locations = num_locations
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx: int):
        random_problem_state = get_random_problem_description(
            num_vehicles=self.num_vehicles,
            num_locations=self.num_locations
        )
        return random_problem_state


def train_one_problem(
        model: ModelBase,
        trainer: ModelBase,
        criterion,
        optimizer: torch.optim.Optimizer,
        problem_description: ProblemDescription
) -> float:
    optimizer.zero_grad()

    problem_state: ProblemState = initialize_problem_state(problem_description=problem_description)

    model.initialize(problem_state)
    trainer.initialize(problem_state)

    total_loss = torch.tensor(0.)

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

        model_action = Action(courier_id=model_courier_id, location_id=model_location_id)
        trainer_action = Action(courier_id=trainer_courier_id, location_id=trainer_location_id)

        model_reward = problem_state.get_reward(action=model_action)
        trainer_reward = problem_state.get_reward(action=trainer_action)

        courier_choice_loss = criterion(model_couriers_logits, torch.tensor(trainer_courier_idx))
        location_choice_loss = criterion(model_locations_logits, torch.tensor(trainer_location_idx))
        total_loss += (courier_choice_loss + location_choice_loss) * (trainer_reward - model_reward)

        problem_state.update(trainer_action)

    total_loss.backward()
    optimizer.step()

    return total_loss.item() / len(problem_description.locations)


def train(
        model: ModelBase,
        trainer: ModelBase,
        criterion,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        dataset
):
    history = defaultdict(lambda: defaultdict(list))

    for epoch in tqdm(range(num_epochs), position=0, leave=True):
        epoch_start_time = time.time()

        problem_losses = []

        for i, problem_description in tqdm(enumerate(dataset), total=len(dataset), position=0, leave=False):
            problem_loss = train_one_problem(
                model=model,
                trainer=trainer,
                criterion=criterion,
                optimizer=optimizer,
                problem_description=problem_description
            )
            history['train']['problem_loss'].append(problem_loss)
            problem_losses.append(problem_loss)
            if i + 1 == len(dataset):
                break

        history['train']['mean_epoch_loss'].append(np.mean(problem_losses))
        history['train']['5th_percentile_epoch_loss'].append(np.quantile(problem_losses, 0.05))
        history['train']['95th_percentile_epoch_loss'].append(np.quantile(problem_losses, 0.95))

        clear_output()
        print(f'Epoch {epoch + 1} of {num_epochs} took {round(time.time() - epoch_start_time, 3)} seconds')
        plot_learning_curves(history)

    return history
