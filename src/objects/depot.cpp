//
// Created by eliseysudakov on 10/2/22.
//

#include "depot.h"
#include <iostream>

void to_json(nlohmann::json &j, const Depot &depot) {
    j = {
            {"id",    depot.id},
            {"point", {
                              {"lat", depot.lat},
                              {"lon", depot.lon},
                      }}
    };
}

void from_json(const nlohmann::json &j, Depot &depot) {
    j.at("id").get_to(depot.id);
    j.at("point").at("lat").get_to(depot.lat);
    j.at("point").at("lon").get_to(depot.lon);
}
