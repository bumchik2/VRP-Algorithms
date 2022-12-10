//
// Created by eliseysudakov on 10/2/22.
//

#pragma once

#include "../objects/problem_objects.h"
#include "../../json/single_include/nlohmann/json.hpp"

#include <string>
#include <unordered_map>
#include <utility>

class TimeMatrix;

class DistanceMatrix {
public:
    DistanceMatrix() = default;  // default constructor for nlohmann::json integrations

    DistanceMatrix(std::unordered_map<std::string, std::unordered_map<std::string, float>>  depots_to_locations_distances,
            std::unordered_map<std::string, std::unordered_map<std::string, float>>  locations_to_locations_distances):
            _depots_to_locations_distances(std::move(depots_to_locations_distances)),
            _locations_to_locations_distances(std::move(locations_to_locations_distances)) {}

    float get_distance_depot_to_location(const std::string& depot_id, const std::string& location_id) const;
    float get_distance_location_to_location(const std::string& location_id_1, const std::string& location_id_2) const;

private:
    std::unordered_map<std::string, std::unordered_map<std::string, float>> _depots_to_locations_distances; // in km
    std::unordered_map<std::string, std::unordered_map<std::string, float>> _locations_to_locations_distances; // in km

    static void _check_positive_distance(const std::string& from_id, const std::string& to_id, float distance);

    friend void to_json(nlohmann::json &j, const DistanceMatrix &distance_matrix);
    friend void from_json(const nlohmann::json &j, DistanceMatrix &distance_matrix);
    friend TimeMatrix get_geodesic_time_matrix(const DistanceMatrix& distance_matrix, const std::string& routing_mode);
};

double to_radians(double degree);

float get_euclidean_distance_km(float lat1, float lon1, float lat2, float lon2);

DistanceMatrix get_euclidean_distance_matrix(const ProblemObjects& problem_objects);

void to_json(nlohmann::json &j, const DistanceMatrix &distance_matrix);

void from_json(const nlohmann::json &j, DistanceMatrix &distance_matrix);
