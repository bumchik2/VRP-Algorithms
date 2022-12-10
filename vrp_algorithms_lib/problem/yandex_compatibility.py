import vrp_algorithms_lib.common_tools.date_helpers as date_helpers
from vrp_algorithms_lib.problem.models import ProblemDescription


def solver_request_to_problem_description(solver_request: dict) -> ProblemDescription:
    # TODO (should be done via c++ script `save_problem_description_to_json`)
    pass


def problem_description_to_solver_request(problem_description: ProblemDescription) -> dict:
    # TODO
    pass


def transform_request(
        solver_request: dict,
        out_of_time_penalty_per_minute: float,
        distance_penalty: float,
        global_proximity_factor: float,
        solver_temperature: int,
        solver_time_limit_s: int = 60
) -> dict:
    """
    Remove fields that are not needed
    :param solver_request: dict with yandex solver request
    :param out_of_time_penalty_per_minute: penalty for one minute of time for each location
    :param distance_penalty: penalty for one km of travel distance
    :param global_proximity_factor: global_proximity_factor
    :param solver_temperature: initial annealing temperature of the solver
    :param solver_time_limit_s: solver_time_limit_s
    (see vrp_algorithms_lib.problem.penalties.global_proximity_penalty_calculator)
    :return: clean dict with yandex solver request
    """
    if 'depot' in solver_request:
        solver_request['depots'] = [solver_request['depot']]
        del solver_request['depot']

    assert len(solver_request['depots']) == 1

    locations = []
    for i, location in enumerate(solver_request['locations']):
        time_window_dict = date_helpers.time_window_str2dict(location['time_window'])
        time_window = f'{date_helpers.seconds_to_time_string(time_window_dict["begin"])}-' \
                      f'{date_helpers.seconds_to_time_string(time_window_dict["end"])}'
        hard_time_window = f'{date_helpers.seconds_to_time_string(time_window_dict["begin"])}-100.00:00:00'
        new_location = {
            'id': f'location {i + 1}',
            'depot_id': 'depot 1',
            'point': location['point'],
            'time_window': time_window,
            'hard_time_window': hard_time_window,
            'penalty': {
                'out_of_time': {
                    'fixed': 0,
                    'minute': out_of_time_penalty_per_minute
                }
            }
        }
        locations.append(new_location)

    vehicles = [
        {
            'id': f'courier {i + 1}',
            'return_to_depot': False,
            'cost': {
                'fixed': 0,
                'hour': 0,
                'km': distance_penalty
            },
            'shifts': [
                {
                    'id': 'shift 1',
                    'time_window': '00:00:00-100.00:00:00'
                }
            ]
        } for i, v in enumerate(solver_request['vehicles'])
    ]

    depots = [
        {
            'id': f'depot {i + 1}',
            'point': d['point'],
            'time_window': '00:00:00-100.00:00:00'
        } for i, d in enumerate(solver_request['depots'])
    ]

    options = {
        'time_zone': 3,
        'matrix_router': 'geodesic',
        'thread_count': 1,
        'task_count': 1,
        'temperature': solver_temperature,
        'global_proximity_factor': global_proximity_factor,
        'solver_time_limit_s': solver_time_limit_s
    }

    new_request = {
        'locations': locations,
        'vehicles': vehicles,
        'depots': depots,
        'options': options
    }

    return new_request
