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

    virtual ~Algorithm() = default;

    virtual void solve_problem() {
        throw std::runtime_error("Calling solve_problem from Algorithm base class");
    };

    void save_penalty(const std::string& filename);
    void save_routes(const std::string& filename);

protected:
    const ProblemDescription& _problem_description;
    ProblemSolution& _problem_solution;
};
