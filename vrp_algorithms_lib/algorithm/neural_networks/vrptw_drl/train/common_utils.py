from typing import Dict
from typing import List
from typing import Optional

import scipy.stats as sps

from vrp_algorithms_lib.problem.models import ProblemDescription, Courier, Location, Depot, DepotId, CourierId, \
    LocationId, Penalties, Point, Routes, DistanceMatrix, TimeMatrix
from vrp_algorithms_lib.problem.models import get_geodesic_time_matrix, get_euclidean_distance_matrix


def common_min_max_value_check(
        num_value: Optional[int],
        min_value: Optional[int],
        max_value: Optional[int],
):
    if min_value is not None:
        assert max_value is not None and num_value is None
    else:
        assert num_value is not None


def get_routes_slice(
        routes: Routes,
        courier_ids: List[CourierId],
        location_ids: Optional[List[LocationId]] = None
) -> Routes:
    routes_filtered = Routes(
        routes=[route for route in routes.routes if route.vehicle_id in courier_ids]
    )

    if location_ids:
        for route in routes_filtered.routes:
            route.location_ids = [location_id for location_id in route.location_ids if location_id in location_ids]

    return routes_filtered


def get_distance_matrix_slice(
        distance_matrix: DistanceMatrix,
        location_ids: List[LocationId]
) -> DistanceMatrix:
    depots_to_locations_distances = {
        depot_id: {
            location_id: distance_matrix.depots_to_locations_distances[depot_id][location_id]
            for location_id in location_ids
        } for depot_id in distance_matrix.depots_to_locations_distances
    }

    assert len(depots_to_locations_distances) == len(distance_matrix.depots_to_locations_distances)
    assert len(list(depots_to_locations_distances.values())[0]) == len(location_ids)

    locations_to_locations_distances = {
        location_id_1: {
            location_id_2: distance_matrix.locations_to_locations_distances[location_id_1][location_id_2]
            for location_id_2 in location_ids
        } for location_id_1 in location_ids
    }

    assert len(locations_to_locations_distances) == len(location_ids)
    assert len(list(locations_to_locations_distances.values())[0]) == len(location_ids)

    return DistanceMatrix(
        depots_to_locations_distances=depots_to_locations_distances,
        locations_to_locations_distances=locations_to_locations_distances
    )


def get_time_matrix_slice(
        time_matrix: TimeMatrix,
        location_ids: List[LocationId]
) -> TimeMatrix:
    depots_to_locations_travel_times = {
        depot_id: {
            location_id: time_matrix.depots_to_locations_travel_times[depot_id][location_id]
            for location_id in location_ids
        } for depot_id in time_matrix.depots_to_locations_travel_times
    }

    assert len(depots_to_locations_travel_times) == len(time_matrix.depots_to_locations_travel_times)
    assert len(list(depots_to_locations_travel_times.values())[0]) == len(location_ids)

    locations_to_locations_travel_times = {
        location_id_1: {
            location_id_2: time_matrix.locations_to_locations_travel_times[location_id_1][location_id_2]
            for location_id_2 in location_ids
        } for location_id_1 in location_ids
    }

    assert len(locations_to_locations_travel_times) == len(location_ids)
    assert len(list(locations_to_locations_travel_times.values())[0]) == len(location_ids)

    return TimeMatrix(
        depots_to_locations_travel_times=depots_to_locations_travel_times,
        locations_to_locations_travel_times=locations_to_locations_travel_times
    )


def get_problem_description_slice(
        problem_description: ProblemDescription,
        courier_ids: List[CourierId],
        location_ids: List[LocationId]
) -> ProblemDescription:
    locations_part = {
        location_id: problem_description.locations[location_id] for location_id in problem_description.locations if
        location_id in location_ids
    }

    couriers_part = {
        courier_id: problem_description.couriers[courier_id] for courier_id in problem_description.couriers if
        courier_id in courier_ids
    }

    distance_matrix = get_distance_matrix_slice(
        problem_description.distance_matrix,
        location_ids=location_ids
    )

    time_matrix = get_time_matrix_slice(
        problem_description.time_matrix,
        location_ids=location_ids
    )

    return ProblemDescription(
        locations=locations_part,
        couriers=couriers_part,
        depots=problem_description.depots,
        distance_matrix=distance_matrix,
        time_matrix=time_matrix,
        penalties=problem_description.penalties
    )


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
            point=Point(
                lat=lat_distribution.rvs(),
                lon=lon_distribution.rvs(),
            )
        )
    }

    locations: Dict[LocationId, Location] = {
        LocationId(f'location {i + 1}'): Location(
            id=LocationId(f'location {i + 1}'),
            depot_id=DepotId('depot 1'),
            point=Point(
                lat=lat_distribution.rvs(),
                lon=lon_distribution.rvs(),
            ),
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
