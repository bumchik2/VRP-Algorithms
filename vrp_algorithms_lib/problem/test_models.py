import os
from pathlib import Path

from vrp_algorithms_lib.common_tools.file_utils import read_json
from vrp_algorithms_lib.problem.models import ProblemDescription, Routes


def general_model_test(relative_path, model_class):
    absolute_path = os.path.normpath(str(Path(__file__).parent / relative_path))
    object_json = read_json(absolute_path)
    model_class.parse_obj(object_json)


def test_problem_description_simple_1():
    general_model_test('../../test_data/inputs/simple_test_1/problem_description.json', ProblemDescription)


def test_problem_description_medium_1():
    general_model_test('../../test_data/inputs/medium_test_1/problem_description.json', ProblemDescription)


def test_routes_simple_1():
    general_model_test('../../test_data/results/annealing/simple_test_1/1000000_iterations_routes_1.json', Routes)


def test_routes_medium_1():
    general_model_test('../../test_data/results/annealing/medium_test_1/1000000_iterations_routes_1.json', Routes)
