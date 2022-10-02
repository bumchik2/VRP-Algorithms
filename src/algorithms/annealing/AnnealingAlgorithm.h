//
// Created by eliseysudakov on 10/2/22.
//

#pragma once

#include "../algorithm.h"
#include "penalties/penalty.h"
#include "penalties/distance_penalty.h"

#include <memory>
#include <vector>

class AnnealingAlgorithm : public Algorithm {
public:
    AnnealingAlgorithm(const ProblemDescription& problem_description, ProblemSolution& problem_solution,
                       int n_iterations, float distance_penalty_multiplier = 0):
            Algorithm(problem_description,problem_solution), _n_iterations(n_iterations) {
        if (distance_penalty_multiplier > 0) {
            _penalties.push_back(std::make_shared<DistancePenalty>(distance_penalty_multiplier));
        }
    }
    void solve_problem() override;

private:
    int _n_iterations;

    std::vector<std::shared_ptr<Penalty>> _penalties;

    void _try_to_apply_random_mutation(int step_number);
};
