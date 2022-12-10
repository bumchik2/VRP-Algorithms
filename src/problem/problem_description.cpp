//
// Created by eliseysudakov on 10/2/22.
//

#include "problem_description.h"
#include "penalties/penalty.h"
#include "penalties/distance_penalty.h"
#include "../utils/files_utils.h"

#include <iostream>


void to_json(nlohmann::json &j, const ProblemDescription &problem_description) {
    j = {
            {"locations",       problem_description.locations},
            {"couriers",        problem_description.couriers},
            {"depots",          problem_description.depots},
            {"distance_matrix", problem_description.distance_matrix},
            {"time_matrix",     problem_description.time_matrix},
            {"penalties",       problem_description.penalties}
    };
}


void from_json(const nlohmann::json &j, ProblemDescription &problem_description) {
    j.at("distance_matrix").get_to(problem_description.distance_matrix);
    j.at("time_matrix").get_to(problem_description.time_matrix);
    j.at("penalties").get_to(problem_description.penalties);
    j.at("couriers").get_to(problem_description.couriers);
    j.at("depots").get_to(problem_description.depots);
    j.at("locations").get_to(problem_description.locations);
}


void save_problem_description_to_json(const ProblemDescription &problem_description, const std::string &filename) {
    const nlohmann::json json_to_save = problem_description;
    save_json(json_to_save, filename);
}
