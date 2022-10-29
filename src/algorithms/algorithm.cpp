//
// Created by eliseysudakov on 10/2/22.
//

#include "algorithm.h"
#include "../utils/files_utils.h"


void Algorithm::save_penalty(const std::string& filename) {
    nlohmann::json json_to_save;
    for (const auto & penalty : _problem_description.penalties) {
        std::string penalty_name = penalty->get_short_name();
        float penalty_value = penalty->get_penalty(_problem_solution.routes);
        json_to_save[penalty_name] = penalty_value;
    }
    save_json(json_to_save, filename);
}


void Algorithm::save_routes(const std::string& filename) {
    nlohmann::json json_to_save = _problem_solution.routes;
    save_json(json_to_save, filename);
}
