//
// Created by eliseysudakov on 10/2/22.
//

#pragma once

#include <string>
#include <unordered_map>
#include <utility>

class DistanceMatrix {
public:
    DistanceMatrix(std::unordered_map<std::string, std::unordered_map<std::string, float>>  depots_to_locations_distances,
            std::unordered_map<std::string, std::unordered_map<std::string, float>>  locations_to_locations_distances):
            _depots_to_locations_distances(std::move(depots_to_locations_distances)),
            _locations_to_locations_distances(std::move(locations_to_locations_distances)) {}

    float get_distance_depot_to_location(const std::string& depot_id, const std::string& location_id) const;
    float get_distance_location_to_location(const std::string& location_id_1, const std::string& location_id_2) const;

private:
    std::unordered_map<std::string, std::unordered_map<std::string, float>> _depots_to_locations_distances;
    std::unordered_map<std::string, std::unordered_map<std::string, float>> _locations_to_locations_distances;

    static void _check_positive_distance(float distance) ;
};
