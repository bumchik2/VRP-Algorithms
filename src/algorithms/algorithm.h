//
// Created by eliseysudakov on 10/2/22.
//

#pragma once

#include "../problem/problem_description.h"
#include "../problem/problem_solution.h"

class Algorithm {
public:
    Algorithm(const ProblemDescription& problem_description, ProblemSolution& problem_solution):
            _problem_description(problem_description), _problem_solution(problem_solution) {}

    virtual void solve_problem() {};

protected:
    const ProblemDescription& _problem_description;
    ProblemSolution& _problem_solution;
};
