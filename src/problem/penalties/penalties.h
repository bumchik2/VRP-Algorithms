//
// Created by eliseysudakov on 11/28/22.
//

#pragma once

#include <vector>
#include <memory>
#include <iostream>

#include "penalty.h"
#include "distance_penalty.h"
#include "global_proximity_penalty.h"
#include "out_of_time_penalty.h"

class Penalties {
public:
    Penalties() = default;

    Penalties(float distance_penalty_multiplier, float global_proximity_factor, float out_of_time_penalty_per_minute) :
            distance_penalty_multiplier(distance_penalty_multiplier), global_proximity_factor(global_proximity_factor),
            out_of_time_penalty_per_minute(out_of_time_penalty_per_minute) {
        initialize();
    }

    void initialize();

    std::vector<std::shared_ptr<Penalty>> penalties{};

    float distance_penalty_multiplier{};
    float global_proximity_factor{};
    float out_of_time_penalty_per_minute{};
};

void to_json(nlohmann::json &j, const Penalties &penalties);

void from_json(const nlohmann::json &j, Penalties &penalties);
