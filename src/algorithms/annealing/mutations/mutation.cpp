//
// Created by eliseysudakov on 10/2/22.
//

#include "mutation.h"
#include "../../../utils/random_utils.h"

void Mutation::set_random_seed(int random_seed) {
    _random_seed = random_seed;
}

void Mutation::_fix_random_seed() const {
    fix_random_seed(_random_seed);
}

std::vector<float> Mutation::_calculate_penalty_part(const ProblemSolution &problem_solution,
                                                     const Penalties &penalties,
                                                     const std::vector<int> &modified_routes_indices) {
    std::vector<float> penalty_part;
    std::vector<Route> modified_routes;
    for (int modified_route_index: modified_routes_indices) {
        modified_routes.push_back(problem_solution.routes[modified_route_index]);
    }

    for (const auto &penalty: penalties.penalties) {
        if (penalty->penalty_type == PER_ROUTE_PENALTY) {
            penalty_part.push_back(penalty->get_penalty(problem_solution.get_problem_description(), modified_routes));
        } else if (penalty->penalty_type == PER_PROBLEM_PENALTY) {
            penalty_part.push_back(
                    penalty->get_penalty(problem_solution.get_problem_description(), problem_solution.routes));
        } else {
            throw std::runtime_error("Unsupported penalty type in _calculate_penalty_part");
        }
    }
    return penalty_part;
}

std::vector<float> Mutation::get_delta_penalties(ProblemSolution &problem_solution,
                                                 const Penalties &penalties) const {
    // If all the penalties are of kind PER_ROUTE_PENALTY, then only modified routes need to be saved
    bool need_to_save_all_the_routes = false;
    for (const auto &penalty: penalties.penalties) {
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
        for (int modified_route_index: modified_routes_indices) {
            old_routes[modified_route_index] = problem_solution.routes[modified_route_index];
        }
    }

    // Calculate initial penalty
    std::vector<float> penalty_parts_initial = _calculate_penalty_part(problem_solution, penalties,
                                                                       modified_routes_indices);

    // Mutate
    mutate(problem_solution);

    // Calculate delta penalty
    std::vector<float> penalty_parts_final = _calculate_penalty_part(problem_solution, penalties,
                                                                     modified_routes_indices);

    // Restore the modified routes
    for (const auto &route_index_to_route: old_routes) {
        problem_solution.routes[route_index_to_route.first] = route_index_to_route.second;
    }

    std::vector<float> delta_penalties;
    for (int i = 0; i < penalty_parts_initial.size(); ++i) {
        delta_penalties.push_back(penalty_parts_final[i] - penalty_parts_initial[i]);
    }
    return delta_penalties;
}
