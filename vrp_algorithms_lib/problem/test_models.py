from vrp_algorithms_lib.problem.test_utils import problem_description_for_tests


def test_problem_state_init(problem_description_for_tests):
    assert problem_description_for_tests.penalties.distance_penalty_multiplier == 8
