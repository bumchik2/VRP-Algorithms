//
// Created by eliseysudakov on 10/2/22.
//

#include "time_matrix.h"

#include <string>

void TimeMatrix::_check_positive_travel_time(float travel_time) const {
    if (travel_time <= 0) {
        std::string error_message = "Zero distance value in TimeMatrix";
        throw std::runtime_error(error_message);
    }
}

float TimeMatrix::get_travel_time_depot_to_location(const std::string& depot_id, const std::string& location_id) const {
    float result = _depots_to_locations_travel_times[depot_id][location_id];
    _check_positive_travel_time(result);
    return result
}

float TimeMatrix::get_travel_time_location_to_location(const std::string& location_id_1, const std::string& location_id_2) const {
    float result = _locations_to_locations_travel_times[location_id_1][location_id_2];
    _check_positive_travel_time(result);
    return result;
}
