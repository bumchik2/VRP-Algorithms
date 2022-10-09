//
// Created by eliseysudakov on 10/2/22.
//

#include "../../json/single_include/nlohmann/json.hpp"

#include "../problem/problem_description.h"

#include <unordered_map>
#include <fstream>
#include <iostream>

using std::cout;
using std::endl;

nlohmann::json read_json(const std::string& path_to_json) {
    std::ifstream read_stream(path_to_json);
    nlohmann::json result;
    result = nlohmann::json::parse(read_stream);
    read_stream.close();
    return result;
}

ProblemDescription read_euclidean_problem(const std::string& test_data_folder) {
    // TODO: это все нужно будет вынести в отдельные функции

    nlohmann::json locations_json = read_json(test_data_folder + "/locations.json");
    nlohmann::json couriers_json = read_json(test_data_folder + "/couriers.json");
    nlohmann::json depots_json = read_json(test_data_folder + "/depots.json");

    std::unordered_map<std::string, Location> locations;
    std::unordered_map<std::string, Courier> couriers;
    std::unordered_map<std::string, Depot> depots;

    for (const auto& location_json: locations_json) {
        locations.insert({
                                 location_json["id"],
                                 Location(
                                         location_json["id"],
                                         location_json["depot_id"],
                                         location_json["lat"],
                                         location_json["lon"],
                                         location_json["time_window_start_s"],
                                         location_json["time_window_end_s"]
                                 )
                         });
    }

    for (const auto& courier_json: couriers_json) {
        couriers.insert({
                                courier_json["id"],
                                Courier(
                                        courier_json["id"],
                                        courier_json["depot_id"]
                                )
                        });
    }

    for (const auto& depot_json: depots_json) {
        depots.insert({
                              depot_json["id"],
                              Depot(
                                      depot_json["id"],
                                      depot_json["lat"],
                                      depot_json["lon"]
                              )
                      });
    }
    std::unordered_map<std::string, std::unordered_map<std::string, float>> depots_to_locations_distances;
    std::unordered_map<std::string, std::unordered_map<std::string, float>> locations_to_locations_distances;

    for (const auto& depot_id_to_depot: depots) {
        for (const auto& location_id_to_location: locations) {
            const Depot& depot = depot_id_to_depot.second;
            const Location& location = location_id_to_location.second;
            auto distance = static_cast<float>(pow(pow(depot.lat - location.lat, 2) + pow(depot.lon - location.lon, 2), 0.5) * 10000);
            depots_to_locations_distances[depot.id][location.id] = distance;
        }
    }

    for (const auto& location_id_to_location1: locations) {
        for (const auto& location_id_to_location2: locations) {
            const Location& location_1 = location_id_to_location1.second;
            const Location& location_2 = location_id_to_location2.second;
            auto distance = static_cast<float>(pow(pow(location_1.lat - location_2.lat, 2) + pow(location_1.lon - location_2.lon, 2), 0.5) * 10000);
            locations_to_locations_distances[location_1.id][location_2.id] = distance;
        }
    }

    DistanceMatrix distance_matrix(depots_to_locations_distances, locations_to_locations_distances);
    TimeMatrix time_matrix({}, {});

    return {locations, couriers, depots, distance_matrix, time_matrix};
}

std::unordered_map<std::string, Location> read_locations_from_json(const std::string& path_to_json) {
    nlohmann::json locations_json = read_json(path_to_json);
    std::cout << locations_json;
    return {};
}
