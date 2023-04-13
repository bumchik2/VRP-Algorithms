import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from IPython.display import clear_output


def get_location_id_to_location(request):
    return {location['id']: location for location in request['locations']}


def plot_route(request: dict, route, ax=None, legend=True):
    if ax is None:
        ax = plt.gca()

    if len(route['location_ids']) == 0:
        return

    location_id_to_location = get_location_id_to_location(request)
    lons = [location_id_to_location[location_id]['point']['lon'] for location_id in route['location_ids']]
    lats = [location_id_to_location[location_id]['point']['lat'] for location_id in route['location_ids']]

    wrapped_depot = [depot for depot in request['depots'] if
                     depot['id'] == location_id_to_location[route['location_ids'][0]]['depot_id']]
    assert len(wrapped_depot) == 1
    depot = wrapped_depot[0]
    depot_lon = depot['point']['lon']
    depot_lat = depot['point']['lat']

    ax.plot([depot_lon] + lons, [depot_lat] + lats, label=route['vehicle_id'])

    if legend:
        ax.legend()


def plot_map(request: dict, ax=None, legend=True):
    need_to_show = (ax is None)
    if ax is None:
        plt.figure(figsize=(12, 8))
        ax = plt.gca()

    ax.set_xlabel('Долгота', fontsize=14)
    ax.set_ylabel('Широта', fontsize=14)

    locations_lons = [location['point']['lon'] for location in request['locations']]
    locations_lats = [location['point']['lat'] for location in request['locations']]
    ax.scatter(locations_lons, locations_lats, c='r', s=10)

    depots_lons = [depot['point']['lon'] for depot in request['depots']]
    depots_lats = [depot['point']['lat'] for depot in request['depots']]
    ax.scatter(depots_lons, depots_lats, c='b', s=100, marker='*', label='depots')

    ax.grid(visible=True)

    if legend:
        ax.legend()

    if need_to_show:
        plt.show()


def plot_routes(request: dict, routes, title='', ax=None, legend=True):
    need_to_show = (ax is None)
    if ax is None:
        plt.figure(figsize=(12, 8))
        ax = plt.gca()

    plot_map(request, ax, legend=legend)

    ax.set_title(title, fontsize=16)

    for route in routes:
        plot_route(request, route, ax, legend=legend)

    if need_to_show:
        plt.show()


def plot_penalty_history(penalty_history, skip_first_n=0, title='', ax=None, legend=True):
    need_to_show = (ax is None)
    if ax is None:
        plt.figure(figsize=(12, 8))
        ax = plt.gca()

    n_iterations = len(penalty_history[list(penalty_history.keys())[0]])

    total_penalty = np.zeros(n_iterations)

    for penalty_type in penalty_history:
        ax.plot(range(skip_first_n, n_iterations), penalty_history[penalty_type][skip_first_n:], label=penalty_type)
        total_penalty += np.array(penalty_history[penalty_type])

    ax.plot(range(skip_first_n, n_iterations), total_penalty[skip_first_n:], label='total-penalty')

    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Iteration number', fontsize=14)
    ax.set_ylabel('Penalty value', fontsize=14)

    ax.grid(visible=True)

    if legend:
        ax.legend(fontsize='x-large')

    if need_to_show:
        plt.show()


def plot_checkpoints(request: dict, checkpoints, legend=True):
    penalty_history_part = defaultdict(list)

    for i, checkpoint in enumerate(checkpoints):
        clear_output(wait=True)

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        fig.suptitle(f'Iteration #{checkpoint["iteration_number"]}', fontsize=20)

        for penalty_type in checkpoint['penalty_values']:
            penalty_history_part[penalty_type].append(checkpoint['penalty_values'][penalty_type])

        plot_routes(request, checkpoint['routes'], ax=axes[0], legend=legend)
        plot_penalty_history(penalty_history_part, ax=axes[1], legend=legend)

        plt.show()
