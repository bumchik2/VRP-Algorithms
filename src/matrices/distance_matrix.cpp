//
// Created by eliseysudakov on 10/2/22.
//

#include "distance_matrix.h"

#include <string>

void DistanceMatrix::_check_positive_distance(float distance) {
    if (distance <= 0) {
        std::string error_message = "Zero distance value in DistanceMatrix";
        throw std::runtime_error(error_message);
    }
}

float DistanceMatrix::get_distance_depot_to_location(const std::string& depot_id, const std::string& location_id) const {
    float result = _depots_to_locations_distances.at(depot_id).at(location_id);
    _check_positive_distance(result);
    return result;
}

float DistanceMatrix::get_distance_location_to_location(const std::string& location_id_1, const std::string& location_id_2) const {
    float result = _locations_to_locations_distances.at(location_id_1).at(location_id_2);
    _check_positive_distance(result);
    return result;
}
