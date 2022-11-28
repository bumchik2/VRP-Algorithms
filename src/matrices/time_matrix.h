//
// Created by eliseysudakov on 10/2/22.
//

#pragma once

#include <string>
#include <unordered_map>
#include "../../json/single_include/nlohmann/json.hpp"

class DistanceMatrix;

class TimeMatrix {
public:
    TimeMatrix(std::unordered_map<std::string, std::unordered_map<std::string, float>>  depots_to_locations_travel_times,
    std::unordered_map<std::string, std::unordered_map<std::string, float>>  locations_to_locations_travel_times):
            _depots_to_locations_travel_times(std::move(depots_to_locations_travel_times)),
            _locations_to_locations_travel_times(std::move(locations_to_locations_travel_times)) {}

    float get_travel_time_depot_to_location(const std::string& depot_id, const std::string& location_id) const;
    float get_travel_time_location_to_location(const std::string& location_id_1, const std::string& location_id_2) const;

private:
    std::unordered_map<std::string, std::unordered_map<std::string, float>> _depots_to_locations_travel_times; // in hours
    std::unordered_map<std::string, std::unordered_map<std::string, float>> _locations_to_locations_travel_times; // in hours

    static void _check_positive_travel_time(const std::string& from_id, const std::string& to_id, float travel_time);

    friend void to_json(nlohmann::json &j, const TimeMatrix &time_matrix);
    friend TimeMatrix get_geodesic_time_matrix(const DistanceMatrix &distance_matrix, const std::string &routing_mode);
};

void to_json(nlohmann::json &j, const TimeMatrix &time_matrix);

TimeMatrix get_geodesic_time_matrix(const DistanceMatrix &distance_matrix, const std::string &routing_mode);
