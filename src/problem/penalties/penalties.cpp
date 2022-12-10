//
// Created by eliseysudakov on 11/28/22.
//

#include "penalties.h"

void to_json(nlohmann::json &j, const Penalties &penalties) {
    j = {
            {"distance_penalty_multiplier", penalties.distance_penalty_multiplier},
            {"global_proximity_factor",     penalties.global_proximity_factor}
    };
}

void from_json(const nlohmann::json &j, Penalties &penalties) {
    j.at("distance_penalty_multiplier").get_to(penalties.distance_penalty_multiplier);
    j.at("global_proximity_factor").get_to(penalties.global_proximity_factor);
}
