//
// Created by eliseysudakov on 10/9/22.
//

#pragma once

#include "algorithm.h"
#include "../../json/single_include/nlohmann/json.hpp"

#include <vector>
#include <stdexcept>
#include <iostream>
#include <unordered_map>
#include <string>

struct CheckPoint {
    int iteration_number;
    std::unordered_map<std::string, float> penalty_values;
    float total_penalty;
    std::vector<Route> routes;
};

void to_json(nlohmann::json& j, const CheckPoint& checkpoint);


class IterativeAlgorithm : public Algorithm {
public:
    IterativeAlgorithm(const ProblemDescription &problem_description, ProblemSolution &problem_solution,
                       int n_iterations, int check_points_number=101) : Algorithm(problem_description, problem_solution),
                                           _n_iterations(n_iterations), _checkpoints_number(check_points_number) {
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

    [[nodiscard]] const std::vector<std::vector<float>>& get_penalty_history() const {
        return _penalty_history;
    }

    void solve_problem() override;

    void save_penalty_history(const std::string& filename) const;

    void save_checkpoints(const std::string& filename) const;

protected:
    std::vector<std::vector<float>> _penalty_history;
    // _penalty_history[i][j] is the value of i-th penalty on the j-th step
    int _n_iterations;

private:
    int _checkpoints_number;
    std::vector<CheckPoint> _checkpoints;
    void _make_checkpoint(int step_number);
};
