from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.test_utils import problem_state_for_tests
from vrp_algorithms_lib.problem.test_utils import problem_description_for_tests
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.objects import ProblemState, Action
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.models.attention_model import AttentionModel
from vrp_algorithms_lib.problem.models import LocationId, CourierId

import pytest
import numpy as np


@pytest.fixture
def attention_model_without_neural_network() -> AttentionModel:
    attention_model = AttentionModel(None)
    return attention_model


def test_get_locations_information(problem_state_for_tests, attention_model_without_neural_network):
    problem_state_for_tests: ProblemState
    attention_model_without_neural_network: AttentionModel
    attention_model_without_neural_network.initialize(problem_state_for_tests, routes=None)

    locations_information = attention_model_without_neural_network.get_locations_information(
        problem_state=problem_state_for_tests
    )

    assert len(locations_information) == len(problem_state_for_tests.problem_description.locations) + 1
    assert locations_information[0][4].item() == 1, 'demand for the first location has to be 1'
    assert locations_information[1][4].item() == 1, 'demand for the second location has to be 1'
    assert locations_information[-1][4].item() == 0, 'demand for the depot has to be 0'

    depot = list(problem_state_for_tests.problem_description.depots.values())[0]
    depot_lat = depot.point.lat
    depot_lon = depot.point.lon
    assert np.isclose(locations_information[-1][0].item(),
                      attention_model_without_neural_network.normalize_lat(depot_lat))
    assert np.isclose(locations_information[-1][1].item(),
                      attention_model_without_neural_network.normalize_lon(depot_lon))

    location_id = problem_state_for_tests.idx_to_location_id[0]
    location = problem_state_for_tests.problem_description.locations[location_id]
    location_lat = location.point.lat
    location_lon = location.point.lon
    assert np.isclose(locations_information[0][0].item(),
                      attention_model_without_neural_network.normalize_lat(location_lat))
    assert np.isclose(locations_information[0][1].item(),
                      attention_model_without_neural_network.normalize_lon(location_lon))


def test_get_vehicles_state_information(problem_state_for_tests, attention_model_without_neural_network):
    problem_state_for_tests: ProblemState
    attention_model_without_neural_network: AttentionModel
    attention_model_without_neural_network.initialize(problem_state_for_tests, routes=None)

    vehicles_state_information = attention_model_without_neural_network.get_vehicles_state_information(
        problem_state=problem_state_for_tests
    )

    depot = list(problem_state_for_tests.problem_description.depots.values())[0]
    depot_lat = depot.point.lat
    depot_lon = depot.point.lon
    assert len(vehicles_state_information) == len(problem_state_for_tests.problem_description.couriers)
    assert np.isclose(vehicles_state_information[0][0].item(),
                      attention_model_without_neural_network.normalize_lat(depot_lat))
    assert np.isclose(vehicles_state_information[0][1].item(),
                      attention_model_without_neural_network.normalize_lon(depot_lon))
    assert vehicles_state_information[0][2].item() == 0,  'normalized number of delivered orders is 0 initially'
    assert vehicles_state_information[0][3].item() == 0,  'normalized travel distance is 0 initially'

    action = Action(location_id=LocationId('location_1'), courier_id=CourierId('courier_1'))
    problem_state_for_tests.update(action)

    vehicles_state_information = attention_model_without_neural_network.get_vehicles_state_information(
        problem_state=problem_state_for_tests
    )
    location_1 = problem_state_for_tests.problem_description.locations[LocationId('location_1')]
    location_1_lat = location_1.point.lat
    location_1_lon = location_1.point.lon
    assert np.isclose(vehicles_state_information[0][0].item(),
                      attention_model_without_neural_network.normalize_lat(location_1_lat))
    assert np.isclose(vehicles_state_information[0][1].item(),
                      attention_model_without_neural_network.normalize_lon(location_1_lon))
