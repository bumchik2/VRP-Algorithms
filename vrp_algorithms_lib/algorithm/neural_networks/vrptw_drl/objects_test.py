from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.test_utils import problem_state_for_tests
from vrp_algorithms_lib.problem.test_utils import problem_description_for_tests
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.objects import ProblemState,\
    Action, DepotId, CourierId, LocationId, extract_routes_from_problem_state, Routes
import numpy as np


def test_problem_state_init(problem_state_for_tests):
    problem_state_for_tests: ProblemState
    assert problem_state_for_tests.problem_description.penalties.distance_penalty_multiplier == 8
    assert problem_state_for_tests.idx_to_location_id[problem_state_for_tests.location_id_to_idx[
        LocationId('location_1')]] == LocationId('location_1')


def test_update(problem_state_for_tests):
    problem_state_for_tests: ProblemState
    action_1 = Action(location_id=LocationId('location_1'), courier_id=CourierId('courier_1'))
    problem_state_for_tests.update(action_1)

    assert len(problem_state_for_tests.vehicle_states) == 1
    vehicle_state = problem_state_for_tests.vehicle_states[0]
    assert vehicle_state.courier_id == CourierId('courier_1')
    assert len(vehicle_state.partial_route) == 2
    assert vehicle_state.total_distance == problem_state_for_tests.problem_description.distance_matrix.\
        depots_to_locations_distances[DepotId('depot_1')][LocationId('location_1')]
    assert problem_state_for_tests.locations_idx == [[len(problem_state_for_tests.problem_description.locations),
                                                      problem_state_for_tests.location_id_to_idx[LocationId('location_1')]]]


def test_extract_routes_from_problem_state(problem_state_for_tests):
    problem_state_for_tests: ProblemState
    action_1 = Action(location_id=LocationId('location_1'), courier_id=CourierId('courier_1'))
    problem_state_for_tests.update(action_1)

    routes: Routes = extract_routes_from_problem_state(problem_state_for_tests)
    route_0 = routes.routes[0]
    assert route_0.vehicle_id == CourierId('courier_1')
    assert len(route_0.location_ids) == 1
    assert route_0.location_ids[0] == LocationId('location_1')


def test_reward(problem_state_for_tests):
    problem_state_for_tests: ProblemState
    action_1 = Action(location_id=LocationId('location_1'), courier_id=CourierId('courier_1'))
    assert problem_state_for_tests.get_reward(action=action_1) == -8 * problem_state_for_tests.problem_description.\
        distance_matrix.depots_to_locations_distances[DepotId('depot_1')][LocationId('location_1')]

    problem_state_for_tests.update(action_1)

    action_2 = Action(location_id=LocationId('location_2'), courier_id=CourierId('courier_1'))
    expected_reward = -149.98911231771075
    assert np.isclose(problem_state_for_tests.get_reward(action=action_2), expected_reward)
