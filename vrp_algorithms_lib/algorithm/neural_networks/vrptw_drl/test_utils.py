import pytest
from vrp_algorithms_lib.problem.test_utils import problem_description_for_tests
from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.objects import ProblemState, initialize_problem_state


@pytest.fixture
def problem_state_for_tests(problem_description_for_tests) -> ProblemState:
    problem_state = initialize_problem_state(problem_description_for_tests)
    return problem_state
