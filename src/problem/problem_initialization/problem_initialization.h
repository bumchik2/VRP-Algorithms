//
// Created by eliseysudakov on 10/2/22.
//

#pragma once

#include "../problem_solution.h"
#include "../problem_description.h"

class ProblemInitialization {
public:
    virtual void initialize(const ProblemDescription& problem_description, ProblemSolution& problem_solution) const {}
};
