//
// Created by eliseysudakov on 11/28/22.
//

#include "penalties.h"

void Penalties::initialize() {
    penalties = {};
    if (distance_penalty_multiplier > 0) {
        penalties.push_back(std::make_shared<DistancePenalty>(distance_penalty_multiplier));
    }
    if (global_proximity_factor > 0 && distance_penalty_multiplier > 0) {
        penalties.push_back(
                std::make_shared<GlobalProximityPenalty>(distance_penalty_multiplier * global_proximity_factor));
    }
    if (out_of_time_penalty_per_minute > 0) {
        penalties.push_back(
                std::make_shared<OutOfTimePenalty>(out_of_time_penalty_per_minute));
    }
}

void to_json(nlohmann::json &j, const Penalties &penalties) {
    j = {
            {"distance_penalty_multiplier", penalties.distance_penalty_multiplier},
            {"global_proximity_factor",     penalties.global_proximity_factor},
            {"out_of_time_penalty_per_minute", penalties.out_of_time_penalty_per_minute}
    };
}

void from_json(const nlohmann::json &j, Penalties &penalties) {
    j.at("distance_penalty_multiplier").get_to(penalties.distance_penalty_multiplier);
    j.at("global_proximity_factor").get_to(penalties.global_proximity_factor);
    j.at("out_of_time_penalty_per_minute").get_to(penalties.out_of_time_penalty_per_minute);
    penalties.initialize();
}
