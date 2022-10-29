import os
from pathlib import Path

from vrp_algorithms_lib.common_tools.file_utils import read_json
from vrp_algorithms_lib.problem.models import ProblemDescription


def general_model_test(problem_description_relative_path):
    problem_description_absolute_path = os.path.normpath(str(Path(__file__).parent / problem_description_relative_path))

    problem_description_json = read_json(problem_description_absolute_path)
    ProblemDescription.parse_obj(problem_description_json)


def test_model_simple_1():
    general_model_test('../../test_data/inputs/simple_test_1/problem_description.json')


def test_model_medium_1():
    general_model_test('../../test_data/inputs/medium_test_1/problem_description.json')
