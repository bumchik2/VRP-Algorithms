//
// Created by eliseysudakov on 10/9/22.
//

#include "iterative_algorithm.h"

void IterativeAlgorithm::_save_checkpoint(int step_number) {
    std::vector<float> penalty_values;
    for (const auto &penalty_ptr: _problem_description.penalties) {
        penalty_values.push_back(penalty_ptr->get_penalty(_problem_solution.routes));
    }

    CheckPoint checkpoint{
            step_number,
            penalty_values,
            _problem_solution.routes
    };

    _checkpoints.push_back(checkpoint);
}

void IterativeAlgorithm::solve_problem() {
    for (int step_number = 0; step_number < _n_iterations; ++step_number) {
        _make_step(step_number);

        if (step_number % (_checkpoints_number - 1) == 0) {
            _save_checkpoint(step_number);
        }
    }
}
