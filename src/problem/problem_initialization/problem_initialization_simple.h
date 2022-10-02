//
// Created by eliseysudakov on 10/2/22.
//

#pragma once

#include "problem_initialization.h"

class ProblemInitializationSimple : public ProblemInitialization {
public:
    void initialize(const ProblemDescription& problem_description, ProblemSolution& problem_solution) const override;
};
