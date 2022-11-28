//
// Created by eliseysudakov on 11/28/22.
//

#pragma once

#include <vector>
#include <memory>

#include "penalty.h"
#include "distance_penalty.h"
#include "global_proximity_penalty.h"

class Penalties {
public:
    Penalties(float distance_penalty_multiplier, float global_proximity_factor) :
            distance_penalty_multiplier(distance_penalty_multiplier), global_proximity_factor(global_proximity_factor) {
        if (distance_penalty_multiplier > 0) {
            penalties.push_back(std::make_shared<DistancePenalty>(distance_penalty_multiplier));
        }
        if (global_proximity_factor > 0) {
            penalties.push_back(
                    std::make_shared<GlobalProximityPenalty>(distance_penalty_multiplier * global_proximity_factor));
        }
    }

    std::vector<std::shared_ptr<Penalty>> penalties{};

    float distance_penalty_multiplier;
    float global_proximity_factor;
};

void to_json(nlohmann::json &j, const Penalties &penalties);