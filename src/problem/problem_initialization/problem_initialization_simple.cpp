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

    for (int i = 0; i < problem_description.locations.size(); ++i) {
        const Location& location = problem_description.locations[i];
        problem_solution.routes[i % problem_description.couriers.size()].location_ids.push_back(location.id);
    }
}
