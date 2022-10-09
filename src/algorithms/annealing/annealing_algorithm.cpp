//
// Created by eliseysudakov on 10/2/22.
//

#include "annealing_algorithm.h"
#include "mutations/mutation.h"
#include "../../utils/random_utils.h"

#include <cmath>
#include <iostream>
#include <numeric>

std::vector<std::shared_ptr<Mutation>> AnnealingAlgorithm::_get_mutations() {
    return {
            std::make_shared<SwapVertexMutation>()
    };
}

std::shared_ptr<Mutation> AnnealingAlgorithm::_choose_random_mutation(
        const std::vector<std::shared_ptr<Mutation>> &mutations) {
    // TODO: there should be different probabilities for different mutations (some of them are more useful, than others)
    return mutations[randint(0, static_cast<int>(mutations.size()))];
}

bool AnnealingAlgorithm::_try_to_apply_random_mutation_and_update_penalties_history(int step_number) {
    // choose random mutation, calculate delta penalty, apply if lucky.
    // in the end update penalties history

    std::shared_ptr<Mutation> mutation = _choose_random_mutation(_mutations);
    mutation->set_random_seed(step_number);

    std::vector<float> delta_penalties = mutation->get_delta_penalties(_problem_solution,
                                                                       _problem_description.penalties);
    float total_delta_penalty = std::accumulate(delta_penalties.begin(), delta_penalties.end(),
                                                decltype(delta_penalties)::value_type(0));

    double mutation_probability = pow(2.71, -total_delta_penalty / _temperature);
    double lucky = random_float();
    bool need_to_apply_mutation = (lucky < mutation_probability);
    if (need_to_apply_mutation) {
        mutation->mutate(_problem_solution);
    }

    for (int i = 0; i < _problem_description.penalties.size(); ++i) {
        float old_penalty_value = _penalty_history[i][_penalty_history[i].size() - 1];
        float delta_penalty_part = 0;
        if (need_to_apply_mutation) {
            delta_penalty_part = delta_penalties[i];
        }
        _penalty_history[i].push_back(old_penalty_value + delta_penalty_part);
    }

    return need_to_apply_mutation;
}

void AnnealingAlgorithm::_update_temperature(int step_number) {
    _temperature = _initial_temperature * (1.0f - static_cast<float>(step_number) / static_cast<float>(_n_iterations));
}

void AnnealingAlgorithm::_make_step(int step_number) {
    _try_to_apply_random_mutation_and_update_penalties_history(step_number);
    _update_temperature(step_number);
}
