//
// Created by eliseysudakov on 10/2/22.
//

#include "swap_vertex_mutation.h"
#include "../../../utils/random_utils.h"

#include <vector>

std::vector<int> SwapVertexMutation::_choose_mutation_parameters_(const ProblemSolution& problem_solution) const {
    _fix_random_seed();
    int route_number = -1, pos_1 = -1, pos_2 = -1;
    while (route_number == -1 || problem_solution.routes[route_number].location_ids.size() <= 2 || pos_1 >= pos_2) {
        route_number = randint(0, static_cast<int>(problem_solution.routes.size()));
        Route route_to_modify = problem_solution.routes[route_number];
        if (route_to_modify.empty()) {
            continue;
        }
        pos_1 = randint(0, static_cast<int>(route_to_modify.location_ids.size()));
        pos_2 = randint(0, static_cast<int>(route_to_modify.location_ids.size()));
    }
    return {route_number, pos_1, pos_2};
}

std::vector<int> SwapVertexMutation::_get_modified_routes_indices(const ProblemSolution &problem_solution) const {
    std::vector<int> parameters = _choose_mutation_parameters_(problem_solution);
    return {parameters[0]};
}

void SwapVertexMutation::mutate(ProblemSolution &problem_solution) const {
    _fix_random_seed();
    std::vector<int> parameters = _choose_mutation_parameters_(problem_solution);
    int route_number = parameters[0];
    int pos_1 = parameters[1];
    int pos_2 = parameters[2];

    Route& route_to_modify = problem_solution.routes[route_number];
    std::swap(route_to_modify.location_ids[pos_1], route_to_modify.location_ids[pos_2]);
}
