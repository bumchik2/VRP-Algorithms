//
// Created by eliseysudakov on 10/2/22.
//

#include "route.h"
#include "../../json/single_include/nlohmann/json.hpp"

void to_json(nlohmann::json &j, const Route &route) {
    j = {
            {"location_ids", route.location_ids},
            {"vehicle_id",   route.vehicle_id}
    };
}