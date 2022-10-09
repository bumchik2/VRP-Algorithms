//
// Created by eliseysudakov on 10/2/22.
//

#pragma once

#include "../../json/single_include/nlohmann/json.hpp"

#include <string>

nlohmann::json read_json(const std::string& path_to_json);

ProblemDescription read_euclidean_problem(const std::string& test_data_folder);
