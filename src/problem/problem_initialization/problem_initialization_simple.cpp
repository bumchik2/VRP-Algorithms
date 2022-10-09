//
// Created by eliseysudakov on 10/2/22.
//

#include "problem_initialization_simple.h"

void ProblemInitializationSimple::initialize(const ProblemDescription& problem_description,
        ProblemSolution& problem_solution) const {
    problem_solution.routes.clear();
    for (const auto& id_to_courier: problem_description.couriers) {
        problem_solution.routes.push_back(Route({{}, id_to_courier.first}));
    }

    int i = 0;
    for (const auto& location_id_to_location : problem_description.locations) {
        const Location& location = location_id_to_location.second;
        problem_solution.routes[i % problem_description.couriers.size()].location_ids.push_back(location.id);
        i += 0;
    }
}
