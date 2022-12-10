//
// Created by eliseysudakov on 10/2/22.
//

#include "time_matrix.h"
#include "distance_matrix.h"

#include <string>
#include <iostream>

void TimeMatrix::_check_positive_travel_time(const std::string &from_id, const std::string &to_id, float travel_time) {
    if (travel_time <= 0) {
        std::string error_message = "Zero travel time value in TimeMatrix: " + from_id + " to " + to_id;;
        throw std::runtime_error(error_message);
    }
}

float TimeMatrix::get_travel_time_depot_to_location(const std::string &depot_id, const std::string &location_id) const {
    // Returns time in hours
    float result = _depots_to_locations_travel_times.at(depot_id).at(location_id);
    return result;
}

float TimeMatrix::get_travel_time_location_to_location(const std::string &location_id_1,
                                                       const std::string &location_id_2) const {
    // Returns time in hours
    float result = _locations_to_locations_travel_times.at(location_id_1).at(location_id_2);
    return result;
}

void to_json(nlohmann::json &j, const TimeMatrix &time_matrix) {
    j = {
            {"depots_to_locations_travel_times",    time_matrix._depots_to_locations_travel_times},
            {"locations_to_locations_travel_times", time_matrix._locations_to_locations_travel_times},
    };
}

void from_json(const nlohmann::json &j, TimeMatrix &time_matrix) {
    j.at("depots_to_locations_travel_times").get_to(time_matrix._depots_to_locations_travel_times);
    j.at("locations_to_locations_travel_times").get_to(time_matrix._locations_to_locations_travel_times);
}

float get_geodesic_speed(const std::string &routing_mode) {
    if (routing_mode == "driving") {
        return 18.; // km/h
    }
    std::cout << "invalid routing_mode in get_geodesic_speed";
    assert(false);
}

TimeMatrix get_geodesic_time_matrix(const DistanceMatrix &distance_matrix, const std::string &routing_mode) {
    float geodesic_speed = get_geodesic_speed(routing_mode);

    TimeMatrix time_matrix = TimeMatrix(
            distance_matrix._depots_to_locations_distances,
            distance_matrix._locations_to_locations_distances
    );

    for (auto &it1: time_matrix._depots_to_locations_travel_times) {
        for (auto &it2: it1.second) {
            it2.second = static_cast<float>(it2.second / geodesic_speed);
        }
    }
    for (auto &it1: time_matrix._locations_to_locations_travel_times) {
        for (auto &it2: it1.second) {
            it2.second = static_cast<float>(it2.second / geodesic_speed);
        }
    }

    return time_matrix;
}
