//
// Created by eliseysudakov on 10/2/22.
//

#include "problem_description.h"
#include "penalties/penalty.h"
#include "penalties/distance_penalty.h"
#include "../utils/files_utils.h"


void to_json(nlohmann::json &j, const ProblemDescription &problem_description) {
    std::unordered_map<std::string, float> penalty_dict;
    penalty_dict["distance_penalty_multiplier"] = problem_description._distance_penalty_multiplier;

    j = {
            {"locations",       problem_description.locations},
            {"couriers",        problem_description.couriers},
            {"depots",          problem_description.depots},
            {"distance_matrix", problem_description.distance_matrix},
            {"time_matrix",     problem_description.time_matrix},
            {"penalties",       penalty_dict}
    };
}


void save_problem_description_to_json(const ProblemDescription &problem_description, const std::string &filename) {
    const nlohmann::json json_to_save = problem_description;
    save_json(json_to_save, filename);
}
