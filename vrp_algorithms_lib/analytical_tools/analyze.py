from typing import Dict, Union, List
from typing import Optional

import numpy as np
import pandas as pd

from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.models.model_metric_calculator import ModelBase, \
    ModelMetricCalculator
from vrp_algorithms_lib.problem.metrics.all_metric_calculators import ALL_METRIC_CALCULATORS
from vrp_algorithms_lib.problem.models import ProblemDescription, Routes
from vrp_algorithms_lib.problem.penalties.total_penalty_calculator import TotalPenaltyCalculator, \
    ALL_PENALTY_CALCULATORS


def get_solution_metrics(
        problem_description: ProblemDescription,
        routes: Routes,
        models: Optional[List[ModelBase]],
        model_names: Optional[List[str]]
) -> Dict[str, Union[float, int]]:
    total_locations_in_routes = sum([len(route.location_ids) for route in routes.routes])
    assert total_locations_in_routes == len(problem_description.locations)

    metrics_dict = {}

    for penalty_calculator in ALL_PENALTY_CALCULATORS + [TotalPenaltyCalculator()]:
        metrics_dict[penalty_calculator.get_penalty_name()] = penalty_calculator.calculate(
            problem_description=problem_description, routes=routes)

    for metric_calculator in ALL_METRIC_CALCULATORS:
        metrics_dict[metric_calculator.get_metric_name()] = metric_calculator.calculate(
            problem_description=problem_description, routes=routes)

    assert (models is None) == (model_names is None)
    if models is not None and model_names is not None:
        assert len(models) == len(model_names)
        for model, model_name in zip(models, model_names):
            metric_calculator_1 = ModelMetricCalculator(model, use_courier_logits=False)
            metric_name_1 = f'{model_name}_metric_without_courier_logits'
            metrics_dict[metric_name_1] = metric_calculator_1.calculate(
                problem_description=problem_description, routes=routes)

            metric_calculator_2 = ModelMetricCalculator(model, use_courier_logits=True)
            metric_name_2 = f'{model_name}_metric_with_courier_logits'
            metrics_dict[metric_name_2] = metric_calculator_2.calculate(
                problem_description=problem_description, routes=routes)

    return metrics_dict


def get_solutions_metrics(
        problem_description_list: List[ProblemDescription],
        routes_list: List[Routes],
        models: Optional[List[ModelBase]],
        model_names: Optional[List[str]]
) -> List[Dict[str, Union[float, int]]]:
    metrics_dicts = []

    for problem_description, routes in zip(problem_description_list, routes_list):
        metrics_dict = get_solution_metrics(problem_description, routes, models, model_names)
        metrics_dicts.append(metrics_dict)

    return metrics_dicts


def get_algorithm_metrics(
        problem_description_list: List[ProblemDescription],
        routes_list: List[Routes],
        models: Optional[List[ModelBase]],
        model_names: Optional[List[str]]
) -> Dict[str, Union[float, int]]:
    solutions_metrics = get_solutions_metrics(problem_description_list, routes_list, models, model_names)

    algorithm_metrics_dict = {}

    for metric_name in solutions_metrics[0].keys():
        for aggregation_function, aggregation_function_name in zip([np.mean, np.median], ['mean', 'median']):
            aggregated_metric_value = aggregation_function([metrics_dict[metric_name]
                                                            for metrics_dict in solutions_metrics])
            algorithm_metrics_dict[f'{aggregation_function_name}_{metric_name}'] = aggregated_metric_value

    return algorithm_metrics_dict


def get_algorithms_comparison_dataframe(
        algorithms_names: List[str],
        problem_description_list: List[ProblemDescription],
        routes_lists: List[List[Routes]],
        models: Optional[List[ModelBase]],
        model_names: Optional[List[str]]
) -> pd.DataFrame:
    records = []

    for algorithm_name, routes_list in zip(algorithms_names, routes_lists):
        algorithm_metrics = get_algorithm_metrics(problem_description_list, routes_list, models, model_names)
        algorithm_metrics['algorithm_name'] = algorithm_name
        records.append(algorithm_metrics)

    return pd.DataFrame(records)


def get_algorithms_comparison_pivot_table(
        algorithms_names: List[str],
        problem_description_list: List[ProblemDescription],
        routes_lists: List[List[Routes]],
        models: Optional[List[ModelBase]],
        model_names: Optional[List[str]]
) -> pd.DataFrame:
    algorithms_comparison_dataframe = get_algorithms_comparison_dataframe(
        algorithms_names,
        problem_description_list,
        routes_lists,
        models,
        model_names
    )

    columns = ['algorithm_name']

    non_metric_columns = ['algorithm_name']
    values = list(set(algorithms_comparison_dataframe.columns) - set(non_metric_columns))

    pivot_table = pd.pivot_table(algorithms_comparison_dataframe, columns=columns, values=values)
    return pivot_table
