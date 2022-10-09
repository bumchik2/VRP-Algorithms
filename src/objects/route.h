//
// Created by eliseysudakov on 10/2/22.
//

#pragma once

#include <vector>
#include <string>

#include "../../json/single_include/nlohmann/json.hpp"

class Route {
public:
    std::vector<std::string> location_ids;
    std::string vehicle_id;

    [[nodiscard]] bool empty() const {
        return location_ids.empty();
    }
};


void to_json(nlohmann::json &j, const Route &route);
