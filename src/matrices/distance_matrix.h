//
// Created by eliseysudakov on 10/2/22.
//

#pragma once

#include "../objects/problem_objects.h"

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

    static void _check_positive_distance(const std::string& from_id, const std::string& to_id, float distance);
};

double to_radians(double degree);

float get_euclidean_distance(float lat1, float lon1, float lat2, float lon2);

DistanceMatrix get_euclidean_distance_matrix(const ProblemObjects& problem_objects);