//
// Created by eliseysudakov on 10/2/22.
//

#pragma once

#include <string>
#include "../../json/single_include/nlohmann/json.hpp"

class Depot {
public:
    Depot() = default;  // default constructor for nlohmann::json integrations

    Depot(std::string id, float lat, float lon) :
            id(std::move(id)), lat(lat), lon(lon) {}

    std::string id;
    float lat = -1.;
    float lon = -1.;
};

void to_json(nlohmann::json &j, const Depot &depot);

void from_json(const nlohmann::json &j, Depot &depot);
