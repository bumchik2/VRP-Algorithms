//
// Created by eliseysudakov on 10/9/22.
//

#include "iterative_algorithm.h"
#include "../../tqdm/tqdm.h"
#include "../utils/files_utils.h"
#include "../objects/route.h"
#include "../../json/single_include/nlohmann/json.hpp"

void to_json(nlohmann::json &j, const CheckPoint &checkpoint) {
    j = {
            {"iteration_number", checkpoint.iteration_number},
            {"penalty_values",   checkpoint.penalty_values},
            {"total_penalty",    checkpoint.total_penalty},
            {"routes",           checkpoint.routes}
    };
}

void IterativeAlgorithm::_make_checkpoint(int step_number) {
    std::unordered_map<std::string, float> penalty_values;
    float total_penalty = 0;
    for (const auto &penalty_ptr: _problem_description.penalties.penalties) {
        float penalty_value = penalty_ptr->get_penalty(_problem_description, _problem_solution.routes);
        penalty_values[penalty_ptr->get_short_name()] = penalty_value;
        total_penalty += penalty_value;
    }

    CheckPoint checkpoint{
            step_number,
            penalty_values,
            total_penalty,
            _problem_solution.routes
    };

    _checkpoints.push_back(checkpoint);
}

void IterativeAlgorithm::solve_problem() {
    int checkpoint_period = _n_iterations / (_checkpoints_number - 1);

    for (auto step_number: tq::trange(_n_iterations)) {
        _make_step(step_number);

        if (step_number % checkpoint_period == 0) {
            _make_checkpoint(step_number);
        }
    }
}

void IterativeAlgorithm::save_penalty_history(const std::string &filename) const {
    // generate json object from _penalty_history and save it to filename
    // penalty history is vector<vector<float>>, where history[i][j] is value for i-th penalty on j-th iteration.
    // the result json format is {penalty1: [val11, val12, val13,...], penalty2: [val21, val22, val23,...], ...}
    nlohmann::json json_to_save;
    for (int i = 0; i < _problem_description.penalties.penalties.size(); ++i) {
        std::string penalty_name = _problem_description.penalties.penalties[i]->get_short_name();
        json_to_save[penalty_name] = _penalty_history[i];
    }
    save_json(json_to_save, filename);
}

void IterativeAlgorithm::save_checkpoints(const std::string &filename) const {
    // generate json object from _checkpoints and save it to checkpoints
    nlohmann::json json_to_save = _checkpoints;
    save_json(json_to_save, filename);
}

void print_penalty_changes(
        const ProblemDescription &problem_description, const IterativeAlgorithm &iterative_algorithm) {
    const auto &penalty_history = iterative_algorithm.get_penalty_history();
    float initial_total_penalty = 0;
    float final_total_penalty = 0;

    for (int i = 0; i < penalty_history.size(); ++i) {
        const auto &penalty = problem_description.penalties.penalties[i];
        const std::string &penalty_name = penalty->get_short_name();
        float initial_value = penalty_history[i][0];
        float final_value = penalty_history[i][penalty_history[i].size() - 1];
        std::cout << std::endl << penalty_name + " change: " << initial_value << " -> " << final_value;

        initial_total_penalty += initial_value;
        final_total_penalty += final_value;
    }

    std::cout << std::endl << "total penalty change:" << initial_total_penalty << " -> " << final_total_penalty;
}
