//
// Created by eliseysudakov on 10/2/22.
//

#pragma once

#include "problem_description.h"
#include "../objects/route.h"

#include <vector>

class ProblemSolution {
public:
    ProblemSolution(ProblemDescription& problem_description): _problem_description(problem_description) {
    }

    const ProblemDescription& get_problem_description() const {
        return _problem_description;
    }

    std::vector<Route> routes;

private:
    const ProblemDescription& _problem_description;
};
