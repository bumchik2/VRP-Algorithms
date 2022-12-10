//
// Created by eliseysudakov on 10/2/22.
//

#pragma once

#include <string>
#include "../../json/single_include/nlohmann/json.hpp"


class Courier {
public:
    Courier() = default;  // default constructor for nlohmann::json integrations

    explicit Courier(std::string id) :
            id(std::move(id)) {}

    std::string id;
};

void to_json(nlohmann::json &j, const Courier &courier);

void from_json(const nlohmann::json &j, Courier &courier);

