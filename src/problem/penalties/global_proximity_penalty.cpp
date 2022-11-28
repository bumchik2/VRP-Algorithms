//
// Created by eliseysudakov on 11/28/22.
//

#include "global_proximity_penalty.h"
#include "../problem_description.h"

#include <string>
#include <iostream>

float GlobalProximityPenalty::get_penalty(const ProblemDescription &problem_description, const std::vector<Route> &routes) const {
    float distance_to_last_point = 0;
    const DistanceMatrix& distance_matrix = problem_description.distance_matrix;

    for (const auto & route : routes) {
        if (route.location_ids.empty()) {
            continue;
        }

        const int route_length = static_cast<int>(route.location_ids.size());
        for (int j = 0; j < route_length - 1; ++j) {
            distance_to_last_point += distance_matrix.get_distance_location_to_location(
                    route.location_ids[j], route.location_ids[route_length - 1]);
        }
    }

    return distance_to_last_point * _penalty_multiplier;
}
