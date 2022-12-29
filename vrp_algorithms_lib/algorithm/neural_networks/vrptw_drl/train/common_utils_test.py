from vrp_algorithms_lib.problem.test_utils import problem_description_for_tests, routes_for_tests
import vrp_algorithms_lib.problem.models as models
import vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.train.common_utils as common_utils


def test_get_routes_slice(routes_for_tests: models.Routes):
    sliced_routes_1 = common_utils.get_routes_slice(routes_for_tests, [models.CourierId('courier_1')])
    assert len(sliced_routes_1.routes) == 1
    assert len(sliced_routes_1.routes[0].location_ids) == 2

    sliced_routes_2 = common_utils.get_routes_slice(routes_for_tests, [models.CourierId('courier_2')])
    assert len(sliced_routes_2.routes) == 0


def test_get_distance_matrix_slice(problem_description_for_tests: models.ProblemDescription):
    distance_matrix_sliced = common_utils.get_distance_matrix_slice(
        problem_description_for_tests.distance_matrix,
        [models.LocationId('location_1')]
    )
    assert len(distance_matrix_sliced.locations_to_locations_distances) == 1
    assert 'location_1' in distance_matrix_sliced.locations_to_locations_distances
    assert list(distance_matrix_sliced.locations_to_locations_distances.keys()) == ['location_1']

    assert len(distance_matrix_sliced.depots_to_locations_distances) == \
           len(problem_description_for_tests.distance_matrix.depots_to_locations_distances)
    assert list(list(distance_matrix_sliced.depots_to_locations_distances.values())[0].keys()) == ['location_1']


def test_get_time_matrix_slice(problem_description_for_tests: models.ProblemDescription):
    time_matrix_sliced = common_utils.get_time_matrix_slice(
        problem_description_for_tests.time_matrix,
        [models.LocationId('location_1')]
    )
    assert len(time_matrix_sliced.locations_to_locations_travel_times) == 1
    assert 'location_1' in time_matrix_sliced.locations_to_locations_travel_times
    assert list(time_matrix_sliced.locations_to_locations_travel_times.keys()) == ['location_1']

    assert len(time_matrix_sliced.depots_to_locations_travel_times) == \
           len(problem_description_for_tests.distance_matrix.depots_to_locations_distances)
    assert list(list(time_matrix_sliced.depots_to_locations_travel_times.values())[0].keys()) == ['location_1']


def test_get_problem_description_slice(problem_description_for_tests: models.ProblemDescription):
    problem_description_slice = common_utils.get_problem_description_slice(
        problem_description_for_tests,
        [models.CourierId('courier_1')],
        [models.LocationId('location_1')]
    )

    assert list(problem_description_slice.locations) == ['location_1']
    assert list(problem_description_slice.couriers) == ['courier_1']
