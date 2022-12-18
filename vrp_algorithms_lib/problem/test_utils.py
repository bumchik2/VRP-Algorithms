import pytest

import vrp_algorithms_lib.problem.models as models


@pytest.fixture
def problem_description_for_tests() -> models.ProblemDescription:
    depots = {
        models.DepotId('depot_1'): models.Depot.parse_obj({
            'id': 'depot_1',
            'lat': 55.752908,
            'lon': 37.624524
        })
    }

    locations = {
        models.LocationId('location_1'): models.Location.parse_obj({
            'id': 'location_1',
            'depot_id': 'depot_1',
            'lat': 55.742940,
            'lon': 37.474298,
            'time_window_start_s': 0,
            'time_window_end_s': 86400
        }),
        models.LocationId('location_2'): models.Location.parse_obj({
            'id': 'location_2',
            'depot_id': 'depot_1',
            'lat': 55.746817,
            'lon': 37.773769,
            'time_window_start_s': 0,
            'time_window_end_s': 86400
        })
    }

    couriers = {
        'courier_1': models.Courier.parse_obj({
            'id': 'courier_1'
        })
    }

    penalties = models.Penalties.parse_obj({
        'distance_penalty_multiplier': 8,
        'global_proximity_factor': 0,
        'out_of_time_penalty_per_minute': 0
    })

    distance_matrix = models.get_euclidean_distance_matrix(depots=depots, locations=locations)
    time_matrix = models.get_geodesic_time_matrix(distance_matrix=distance_matrix, depots=depots, locations=locations)

    return models.ProblemDescription(
        locations=locations,
        depots=depots,
        couriers=couriers,
        penalties=penalties,
        distance_matrix=distance_matrix,
        time_matrix=time_matrix
    )
