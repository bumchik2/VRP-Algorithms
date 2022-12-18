from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.test_utils import problem_state_for_tests
from vrp_algorithms_lib.problem.test_utils import problem_description_for_tests
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.objects import ProblemState
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.models.attention_model import AttentionModel

import pytest


@pytest.fixture
def attention_model_without_neural_network() -> AttentionModel:
    attention_model = AttentionModel(None)
    return attention_model


def test_get_locations_information(problem_state_for_tests, attention_model_without_neural_network):
    problem_state_for_tests: ProblemState
    attention_model_without_neural_network: AttentionModel
    attention_model_without_neural_network.initialize(problem_state_for_tests)

    locations_information = attention_model_without_neural_network.get_locations_information(
        problem_state=problem_state_for_tests
    )

    assert len(locations_information) == len(problem_state_for_tests.problem_description.locations) + 1
    assert locations_information[0][4].item() == 1, 'demand for the first location has to be 1'
    assert locations_information[1][4].item() == 1, 'demand for the second location has to be 1'
    assert locations_information[2][4].item() == 0, 'demand for the depot has to be 0'


def test_get_vehicles_state_information(problem_state_for_tests, attention_model_without_neural_network):
    problem_state_for_tests: ProblemState
    attention_model_without_neural_network: AttentionModel
    attention_model_without_neural_network.initialize(problem_state_for_tests)

    vehicles_state_information = attention_model_without_neural_network.get_vehicles_state_information(
        problem_state=problem_state_for_tests
    )

    assert len(vehicles_state_information) == len(problem_state_for_tests.problem_description.couriers)
    assert vehicles_state_information[0][2].item() == 0,  'normalized number of delivered orders is 0 initially'
    assert vehicles_state_information[0][3].item() == 0,  'normalized travel distance is 0 initially'
