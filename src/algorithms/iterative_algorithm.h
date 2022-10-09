//
// Created by eliseysudakov on 10/9/22.
//

#pragma once

#include "algorithm.h"

#include <vector>
#include <stdexcept>

class IterativeAlgorithm : public Algorithm {
public:
    IterativeAlgorithm(const ProblemDescription &problem_description, ProblemSolution &problem_solution,
                       int n_iterations) : Algorithm(problem_description, problem_solution),
                                           _n_iterations(n_iterations) {
        // initialize penalty history
        _penalty_history.resize(problem_description.penalties.size());
        for (int i = 0; i < problem_description.penalties.size(); ++i) {
            const std::shared_ptr<Penalty> penalty_ptr = problem_description.penalties[i];
            _penalty_history[i].push_back(penalty_ptr->get_penalty(problem_solution.routes));
        }
    }

    virtual void _make_step(int step_number) {
        // function that makes one step of iterative algorithm (includes updating _penalty_history)
        throw std::runtime_error("virtual _make_step from class IterativeAlgorithm is called");
    };

    const std::vector<std::vector<float>>& get_history() const {
        return _penalty_history;
    }

    void solve_problem() override;

protected:
    std::vector<std::vector<float>> _penalty_history;
    // _penalty_history[i][j] is the value of i-th penalty on the j-th step
    int _n_iterations;
};
