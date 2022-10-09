//
// Created by eliseysudakov on 10/2/22.
//

#pragma once

#include "../algorithm.h"
#include "../iterative_algorithm.h"
#include "../../problem/penalties/penalty.h"
#include "../../problem/penalties/distance_penalty.h"
#include "mutations/mutation.h"
#include "mutations/swap_vertex_mutation.h"

#include <memory>
#include <vector>

class AnnealingAlgorithm : public IterativeAlgorithm {
public:
    AnnealingAlgorithm(const ProblemDescription &problem_description, ProblemSolution &problem_solution,
                       int n_iterations, float initial_temperature) :
            IterativeAlgorithm(problem_description, problem_solution, n_iterations),
            _initial_temperature(initial_temperature), _temperature(_initial_temperature),
            _mutations(_get_mutations()) {}

    ~AnnealingAlgorithm() override = default;

    void solve_problem() override;

private:
    float _initial_temperature;
    float _temperature;
    std::vector<std::shared_ptr<Mutation>> _mutations{};

    void _make_step(int step_number) override;

    static std::vector<std::shared_ptr<Mutation>> _get_mutations(); // TODO: parametrize the mutations to use

    static std::shared_ptr<Mutation> _choose_random_mutation(const std::vector<std::shared_ptr<Mutation>> &mutations);

    bool _try_to_apply_random_mutation_and_update_penalties_history(int step_number);

    void _update_temperature(int step_number);
};
