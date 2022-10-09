//
// Created by eliseysudakov on 10/2/22.
//

#include "distance_penalty.h"
#include "../problem_description.h"

#include <string>
#include <iostream>

float DistancePenalty::get_penalty(const std::vector<Route> &routes) const {
    float total_distance = 0;
    const DistanceMatrix& distance_matrix = _problem_description.distance_matrix;

    for (const auto & route : routes) {
        if (route.location_ids.empty()) {
            continue;
        }

        std::string first_location_id = route.location_ids[0];
        std::string depot_id = _problem_description.locations.at(first_location_id).depot_id;
        total_distance += distance_matrix.get_distance_depot_to_location(depot_id, first_location_id);

        for (int j = 0; j < route.location_ids.size() - 1; ++j) {
            total_distance += distance_matrix.get_distance_location_to_location(
        route.location_ids[j], route.location_ids[j + 1]);
        }
    }

    return total_distance * _penalty_multiplier;
}
