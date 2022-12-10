import os
import subprocess
from typing import Union

import vrp_algorithms_lib.common_tools.file_utils as file_utils
from vrp_algorithms_lib.algorithm.base_algorithm import BaseAlgorithm
from vrp_algorithms_lib.problem.models import ProblemDescription
from vrp_algorithms_lib.problem.models import Routes


class AnnealingAlgorithm(BaseAlgorithm):
    def __init__(
            self,
            path_to_annealing_binary: Union[str, os.PathLike],
            tmp_path_for_problem_description: Union[str, os.PathLike],
            tmp_path_for_routes: Union[str, os.PathLike],
            n_iterations: int,
            initial_temperature: Union[int, float],
            verbose: bool = False
    ):
        self.path_to_annealing_binary = path_to_annealing_binary
        self.tmp_path_for_problem_description = tmp_path_for_problem_description
        self.tmp_path_for_routes = tmp_path_for_routes
        self.n_iterations = n_iterations
        self.initial_temperature = initial_temperature
        self.verbose = verbose

    def solve_problem(
            self,
            problem_description: ProblemDescription,
            need_to_save_tmp_problem_description: bool = True,
            remove_tmp_problem_description: bool = True,
            remove_tmp_routes: bool = True
    ) -> Routes:
        python_path_to_problem_description = self.tmp_path_for_problem_description
        python_path_to_routes = self.tmp_path_for_routes

        if self.verbose:
            print(f'python_path_to_problem_description = {python_path_to_problem_description}')
            print(f'python_path_to_routes = {python_path_to_routes}')

        if need_to_save_tmp_problem_description:
            file_utils.save_json(problem_description.dict(), python_path_to_problem_description, indent=None)

        args = [
            f'./{self.path_to_annealing_binary}',
            self.tmp_path_for_problem_description,
            self.tmp_path_for_routes,
            self.n_iterations,
            self.initial_temperature
        ]
        args = [str(arg) for arg in args]

        # Run annealing binary, save json with routes to tmp path for routes
        if self.verbose:
            print(f'Running the following command: {args}')
        subprocess.check_output(args)

        routes: Routes = Routes.parse_obj(file_utils.read_json(python_path_to_routes))

        if remove_tmp_problem_description:
            os.remove(python_path_to_problem_description)
        if remove_tmp_routes:
            os.remove(python_path_to_routes)

        return routes
