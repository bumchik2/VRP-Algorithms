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
