//
// Created by eliseysudakov on 10/2/22.
//

#include "annealing_algorithm.h"
#include "mutations/mutation.h"
#include "../../utils/random_utils.h"

#include <cmath>

std::vector<std::shared_ptr<Mutation>> AnnealingAlgorithm::_get_mutations(int step_number) {
    return {
        std::make_shared<SwapVertexMutation>(step_number)
    };
}

std::shared_ptr<Mutation> AnnealingAlgorithm::_choose_random_mutation(
        const std::vector<std::shared_ptr<Mutation>>& mutations) {
    return mutations[randint(0, static_cast<int>(mutations.size()))];
}

bool AnnealingAlgorithm::_try_to_apply_random_mutation(int step_number) const {
    // choose random mutation, calculate delta penalty, apply if lucky
    // TODO: mutations need to be created once and then random seeds need to be changed
    std::vector<std::shared_ptr<Mutation>> mutations = _get_mutations(step_number);

    std::shared_ptr<Mutation> mutation = _choose_random_mutation(mutations);

    float delta_penalty = mutation->get_delta_penalty(_problem_solution, _penalties);

    double mutation_probability = pow(2.71, -delta_penalty / _temperature);
    double lucky = random_float();
    if (lucky < mutation_probability) {
        mutation->mutate(_problem_solution);
        return true;
    }
    return false;
}

void AnnealingAlgorithm::_update_temperature(int step_number) {
    _temperature = _initial_temperature * (1.0f - static_cast<float>(step_number) / static_cast<float>(_n_iterations));
}

void AnnealingAlgorithm::solve_problem() {
    for (int step_number = 0; step_number < _n_iterations; ++step_number) {
        _try_to_apply_random_mutation(step_number);
        _update_temperature(step_number);
    }
}
