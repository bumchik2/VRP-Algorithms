//
// Created by eliseysudakov on 10/2/22.
//

#pragma once

#include <string>
#include <unordered_map>

class TimeMatrix {
public:
    TimeMatrix(std::unordered_map<std::string, std::unordered_map<std::string, float>>  depots_to_locations_travel_times,
    std::unordered_map<std::string, std::unordered_map<std::string, float>>  locations_to_locations_travel_times):
            _depots_to_locations_travel_times(std::move(depots_to_locations_travel_times)),
            _locations_to_locations_travel_times(std::move(locations_to_locations_travel_times)) {}

    float get_travel_time_depot_to_location(const std::string& depot_id, const std::string& location_id) const;
    float get_travel_time_location_to_location(const std::string& location_id_1, const std::string& location_id_2) const;

private:
    std::unordered_map<std::string, std::unordered_map<std::string, float>> _depots_to_locations_travel_times;
    std::unordered_map<std::string, std::unordered_map<std::string, float>> _locations_to_locations_travel_times;

    static void _check_positive_travel_time(float travel_time) ;
};

