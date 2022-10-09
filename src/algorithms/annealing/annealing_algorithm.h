//
// Created by eliseysudakov on 10/2/22.
//

#pragma once

#include "../algorithm.h"
#include "penalties/penalty.h"
#include "penalties/distance_penalty.h"
#include "mutations/mutation.h"
#include "mutations/swap_vertex_mutation.h"

#include <memory>
#include <vector>

class AnnealingAlgorithm : public Algorithm {
public:
    AnnealingAlgorithm(const ProblemDescription& problem_description, ProblemSolution& problem_solution,
                       int n_iterations, float initial_temperature, float distance_penalty_multiplier = 0):
            Algorithm(problem_description,problem_solution), _n_iterations(n_iterations),
            _initial_temperature(initial_temperature), _temperature(_initial_temperature) {
        if (distance_penalty_multiplier > 0) {
            _penalties.push_back(std::make_shared<DistancePenalty>(distance_penalty_multiplier));
        }
    }

    ~AnnealingAlgorithm() override = default;

    void solve_problem() override;

private:
    int _n_iterations;
    float _initial_temperature;
    float _temperature;

    std::vector<std::shared_ptr<Penalty>> _penalties;
    static std::vector<std::shared_ptr<Mutation>> _get_mutations(int step_number);

    static std::shared_ptr<Mutation> _choose_random_mutation(const std::vector<std::shared_ptr<Mutation>>& mutations) ;
    bool _try_to_apply_random_mutation(int step_number) const;

    void _update_temperature(int step_number);
};
