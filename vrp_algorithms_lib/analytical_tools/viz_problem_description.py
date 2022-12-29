import matplotlib.pyplot as plt

from vrp_algorithms_lib.problem.models import ProblemDescription, Route, Routes


def plot_route(problem_description: ProblemDescription, route: Route, ax=None, legend: bool = True):
    need_to_show = (ax is None)

    if ax is None:
        ax = plt.gca()

    if len(route.location_ids) == 0:
        return

    wrapped_depot = list(problem_description.depots.values())
    assert len(wrapped_depot) == 1
    depot = wrapped_depot[0]
    depot_lon = depot.point.lon
    depot_lat = depot.point.lat

    lons = [problem_description.locations[location_id].point.lon for location_id in route.location_ids]
    lats = [problem_description.locations[location_id].point.lat for location_id in route.location_ids]

    ax.plot([depot_lon] + lons, [depot_lat] + lats, label=route.vehicle_id)

    if legend:
        ax.legend()

    if need_to_show:
        plt.show()


def plot_map(problem_description: ProblemDescription, ax=None, legend: bool = True):
    need_to_show = (ax is None)

    if ax is None:
        plt.figure(figsize=(12, 8))
        ax = plt.gca()

    ax.set_xlabel('longitude', fontsize=14)
    ax.set_ylabel('latitude', fontsize=14)

    lons = [location.point.lon for location in problem_description.locations.values()]
    lats = [location.point.lat for location in problem_description.locations.values()]
    ax.scatter(lons, lats, c='r', s=10)

    depots_lons = [depot.point.lon for depot in problem_description.depots.values()]
    depots_lats = [depot.point.lat for depot in problem_description.depots.values()]
    ax.scatter(depots_lons, depots_lats, c='b', s=100, marker='*', label='depots')

    ax.grid(visible=True)

    if legend:
        ax.legend()

    if need_to_show:
        plt.show()


def plot_routes(problem_description: ProblemDescription, routes: Routes, title='', ax=None, legend: bool = True):
    need_to_show = (ax is None)
    if ax is None:
        plt.figure(figsize=(12, 8))
        ax = plt.gca()

    plot_map(problem_description, ax, legend=legend)

    ax.set_title(title, fontsize=16)

    for route in routes.routes:
        plot_route(problem_description, route, ax, legend=legend)

    if need_to_show:
        plt.show()
