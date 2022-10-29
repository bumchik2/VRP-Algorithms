//
// Created by eliseysudakov on 10/2/22.
//

#pragma once

#include <string>
#include "../../json/single_include/nlohmann/json.hpp"


class Courier {
public:
    Courier(std::string id) :
            id(std::move(id)) {}

    std::string id;
};

void to_json(nlohmann::json &j, const Courier& courier);
