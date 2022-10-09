//
// Created by eliseysudakov on 10/9/22.
//

#include "iterative_algorithm.h"

void IterativeAlgorithm::solve_problem() {
    for (int step_number = 0; step_number < _n_iterations; ++step_number) {
        _make_step(step_number);
    }
}
