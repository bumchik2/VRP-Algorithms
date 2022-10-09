//
// Created by eliseysudakov on 10/2/22.
//

#pragma once

#include "../../json/single_include/nlohmann/json.hpp"

#include "../problem/problem_description.h"
#include "../objects/problem_objects.h"

#include <string>

nlohmann::json read_json(const std::string &path_to_json);
void save_json(const nlohmann::json& json_to_save, const std::string& filename);

ProblemObjects read_request(const std::string &path_to_request);

ProblemDescription read_euclidean_problem(const std::string &path_to_request);
