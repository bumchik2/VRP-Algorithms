//
// Created by eliseysudakov on 10/16/22.
//

#include "transfer_mutation.h"
#include "../../../utils/random_utils.h"

#include <vector>
#include <iostream>

std::vector<int> TransferMutation::_choose_mutation_parameters_(const ProblemSolution &problem_solution) const {
    if (problem_solution.routes.size() <= 1) {
        return {};
    }

    _fix_random_seed();
    int route_number_1 = -1, route_number_2 = -1, pos_1 = -1, pos_2 = -1;
    while (route_number_1 == -1 || problem_solution.routes[route_number_1].empty()) {
        route_number_1 = randint(0, static_cast<int>(problem_solution.routes.size()));
        route_number_2 = (route_number_1 + randint(1, static_cast<int>(problem_solution.routes.size()))) %
                         static_cast<int>(problem_solution.routes.size());
        if (problem_solution.routes[route_number_1].empty()) {
            continue;
        }
        pos_1 = randint(0, static_cast<int>(problem_solution.routes[route_number_1].location_ids.size()));
        pos_2 = randint(0, static_cast<int>(problem_solution.routes[route_number_2].location_ids.size() + 1));
    }

    return std::vector<int>{route_number_1, route_number_2, pos_1, pos_2};
}

std::vector<int> TransferMutation::_get_modified_routes_indices(const ProblemSolution &problem_solution) const {
    if (problem_solution.routes.size() <= 1) {
        return {};
    }

    std::vector<int> parameters = _choose_mutation_parameters_(problem_solution);
    return {parameters[0], parameters[1]};
}

void TransferMutation::mutate(ProblemSolution &problem_solution) const {
    if (problem_solution.routes.size() <= 1) {
        return;
    }

    _fix_random_seed();
    std::vector<int> parameters = _choose_mutation_parameters_(problem_solution);
    int route_number_1 = parameters[0];
    int route_number_2 = parameters[1];
    int pos_1 = parameters[2];
    int pos_2 = parameters[3];

    // remove the location from route_1
    std::vector<std::string>& location_ids_1 = problem_solution.routes[route_number_1].location_ids;
    const std::string transferred_location_id = location_ids_1[pos_1];
    location_ids_1.erase(location_ids_1.begin() + pos_1);

    // insert the location into route_2
    std::vector<std::string>& location_ids_2 = problem_solution.routes[route_number_2].location_ids;
    location_ids_2.insert(location_ids_2.begin() + pos_2, transferred_location_id);
}
