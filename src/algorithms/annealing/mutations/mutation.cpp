//
// Created by eliseysudakov on 10/2/22.
//

#include "mutation.h"
#include "../../../utils/random_utils.h"
#include "../penalties/penalty.h"

void Mutation::_fix_random_seed() const {
    fix_random_seed(_random_seed);
}


float Mutation::_calculate_penalty_part(const ProblemSolution& problem_solution,
                                        const std::vector<std::shared_ptr<Penalty>>& penalties,
                                        const std::vector<int>& modified_routes_indices) {
    float penalty_part = 0;
    std::vector<Route> modified_routes;
    for (int modified_route_index : modified_routes_indices) {
        modified_routes.push_back(problem_solution.routes[modified_route_index]);
    }

    for (const auto& penalty: penalties) {
        if (penalty->penalty_type == PER_ROUTE_PENALTY) {
            penalty_part += penalty->get_penalty(problem_solution.get_problem_description(), modified_routes);
        } else if (penalty->penalty_type == PER_PROBLEM_PENALTY) {
            penalty_part += penalty->get_penalty(problem_solution.get_problem_description(), problem_solution.routes);
        } else {
            throw std::runtime_error("Unsupported penalty type in _calculate_penalty_part");
        }
    }
    return penalty_part;
}

float Mutation::get_delta_penalty(ProblemSolution& problem_solution,
            const std::vector<std::shared_ptr<Penalty>>& penalties) const {
    // If all the penalties are of kind PER_ROUTE_PENALTY, then only modified routes need to be saved
    bool need_to_save_all_the_routes = false;
    for (const auto& penalty: penalties) {
        if (penalty->penalty_type != PER_ROUTE_PENALTY) {
            need_to_save_all_the_routes = true;
            break;
        }
    }

    // Save all the routes that influence delta penalty
    const std::vector<int> modified_routes_indices = _get_modified_routes_indices(problem_solution);
    std::unordered_map<int, Route> old_routes;
    if (need_to_save_all_the_routes) {
        for (int i = 0; i < problem_solution.routes.size(); ++i) {
            old_routes[i] = problem_solution.routes[i];
        }
    } else {
        for (int modified_route_index : modified_routes_indices) {
            old_routes[modified_route_index] = problem_solution.routes[modified_route_index];
        }
    }

    // Calculate initial penalty
    float penalty_part_initial = _calculate_penalty_part(problem_solution, penalties, modified_routes_indices);

    // Mutate
    mutate(problem_solution);

    // Calculate delta penalty
    float penalty_part_final = _calculate_penalty_part(problem_solution, penalties, modified_routes_indices);

    // Restore the modified routes
    for (const auto& route_index_to_route: old_routes) {
        problem_solution.routes[route_index_to_route.first] = route_index_to_route.second;
    }
    return penalty_part_final - penalty_part_initial;
}
