//
// Created by eliseysudakov on 10/2/22.
//

#include "AnnealingAlgorithm.h"

void AnnealingAlgorithm::_try_to_apply_random_mutation(int step_number) {
    // choose random mutation, calculate delta penalty, apply if lucky
}

void AnnealingAlgorithm::solve_problem() {
    for (int step_number = 0; step_number < _n_iterations; ++step_number) {
        _try_to_apply_random_mutation(step_number);
    }
}
