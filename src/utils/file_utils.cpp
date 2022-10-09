//
// Created by eliseysudakov on 10/2/22.
//

#include "../../json/single_include/nlohmann/json.hpp"
#include "../problem/problem_description.h"
#include "files_utils.h"
#include "time_utils.h"

#include <unordered_map>
#include <fstream>
#include <iostream>

using std::cout;
using std::endl;


nlohmann::json read_json(const std::string &path_to_json) {
    std::ifstream read_stream(path_to_json);
    nlohmann::json result;
    result = nlohmann::json::parse(read_stream);
    read_stream.close();
    return result;
}

void save_json(const nlohmann::json& json_to_save, const std::string& filename) {
    std::ofstream file(filename);
    file << json_to_save;
}

ProblemObjects read_request(const std::string &path_to_request) {
    std::unordered_map<std::string, Location> locations;
    std::unordered_map<std::string, Courier> couriers;
    std::unordered_map<std::string, Depot> depots;

    nlohmann::json request_json = read_json(path_to_request);

    for (const auto &location_json: request_json["locations"]) {
        const auto time_window_parsed = time_window_to_begin_seconds_end_seconds(location_json["time_window"]);
        locations.insert({
                                 location_json["id"],
                                 Location(
                                         location_json["id"],
                                         location_json["depot_id"],
                                         location_json["point"]["lat"],
                                         location_json["point"]["lon"],
                                         time_window_parsed.first,
                                         time_window_parsed.second
                                 )
                         });
    }

    for (const auto &courier_json: request_json["vehicles"]) {
        couriers.insert({
                                courier_json["id"],
                                Courier(
                                        courier_json["id"]
                                )
                        });
    }

    for (const auto &depot_json: request_json["depots"]) {
        depots.insert({
                              depot_json["id"],
                              Depot(
                                      depot_json["id"],
                                      depot_json["point"]["lat"],
                                      depot_json["point"]["lon"]
                              )
                      });
    }

    return {locations, couriers, depots};
}

ProblemDescription read_euclidean_problem(const std::string &path_to_request) {
    ProblemObjects problem_objects = read_request(path_to_request);
    DistanceMatrix distance_matrix = get_euclidean_distance_matrix(problem_objects);
    // TODO: load time matrices somehow
    TimeMatrix time_matrix({}, {});
    // TODO: distance_penalty_multiplier should be passed from outside
    return {problem_objects.locations, problem_objects.couriers, problem_objects.depots, distance_matrix, time_matrix,
            10};
}
