//
// Created by eliseysudakov on 10/2/22.
//

#include "distance_matrix.h"

#include <string>
#include <algorithm>
#include <cmath>
#include <iostream>

void DistanceMatrix::_check_positive_distance(const std::string& from_id, const std::string& to_id, float distance) {
    if (distance <= 0) {
        std::string error_message = "Zero distance value in DistanceMatrix: " + from_id + " to " + to_id;
        throw std::runtime_error(error_message);
    }
}

float
DistanceMatrix::get_distance_depot_to_location(const std::string &depot_id, const std::string &location_id) const {
    float result = _depots_to_locations_distances.at(depot_id).at(location_id);
    return result;
}

float DistanceMatrix::get_distance_location_to_location(const std::string &location_id_1,
                                                        const std::string &location_id_2) const {
    float result = _locations_to_locations_distances.at(location_id_1).at(location_id_2);
    return result;
}

double to_radians(double degree) {
    return degree / 180 * 3.14159265358979323846;
}

float get_euclidean_distance_km(float lat1, float lon1, float lat2, float lon2) {
    if (lat1 == lat2 and lon1 == lon2) {
        return 0;
    }

    double dist;
    dist = sin(to_radians(lat1)) * sin(to_radians(lat2)) +
           cos(to_radians(lat1)) * cos(to_radians(lat2)) * cos(to_radians(lon1 - lon2));
    dist = acos(dist);
    dist = 6371 * dist;
    return static_cast<float>(dist);
}

DistanceMatrix get_euclidean_distance_matrix(const ProblemObjects &problem_objects) {
    std::unordered_map<std::string, std::unordered_map<std::string, float>> depots_to_locations_distances;
    std::unordered_map<std::string, std::unordered_map<std::string, float>> locations_to_locations_distances;

    for (const auto &depot_id_to_depot: problem_objects.depots) {
        for (const auto &location_id_to_location: problem_objects.locations) {
            const Depot &depot = depot_id_to_depot.second;
            const Location &location = location_id_to_location.second;
            auto distance = get_euclidean_distance_km(depot.lat, depot.lon, location.lat, location.lon);
            depots_to_locations_distances[depot.id][location.id] = distance;
        }
    }

    for (const auto &location_id_to_location1: problem_objects.locations) {
        for (const auto &location_id_to_location2: problem_objects.locations) {
            const Location &location_1 = location_id_to_location1.second;
            const Location &location_2 = location_id_to_location2.second;
            auto distance = get_euclidean_distance_km(location_1.lat, location_1.lon, location_2.lat, location_2.lon);
            locations_to_locations_distances[location_1.id][location_2.id] = distance;
        }
    }

    return {depots_to_locations_distances, locations_to_locations_distances};
}

void to_json(nlohmann::json &j, const DistanceMatrix &distance_matrix) {
    j = {
            {"depots_to_locations_distances", distance_matrix._depots_to_locations_distances},
            {"locations_to_locations_distances", distance_matrix._locations_to_locations_distances},
    };
}
