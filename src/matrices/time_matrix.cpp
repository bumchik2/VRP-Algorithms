//
// Created by eliseysudakov on 10/2/22.
//

#include "time_matrix.h"

#include <string>

void TimeMatrix::_check_positive_travel_time(const std::string& from_id, const std::string& to_id, float travel_time) {
    if (travel_time <= 0) {
        std::string error_message = "Zero travel time value in TimeMatrix: " + from_id + " to " + to_id;;
        throw std::runtime_error(error_message);
    }
}

float TimeMatrix::get_travel_time_depot_to_location(const std::string& depot_id, const std::string& location_id) const {
    float result = _depots_to_locations_travel_times.at(depot_id).at(location_id);
    return result;
}

float TimeMatrix::get_travel_time_location_to_location(const std::string& location_id_1, const std::string& location_id_2) const {
    float result = _locations_to_locations_travel_times.at(location_id_1).at(location_id_2);
    return result;
}

void to_json(nlohmann::json &j, const TimeMatrix &time_matrix) {
    j = {
            {"depots_to_locations_travel_times", time_matrix._depots_to_locations_travel_times},
            {"locations_to_locations_travel_times", time_matrix._locations_to_locations_travel_times},
    };
}
