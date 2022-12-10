//
// Created by eliseysudakov on 10/2/22.
//

#include "../../json/single_include/nlohmann/json.hpp"
#include "../problem/problem_description.h"
#include "../matrices/time_matrix.h"

#include "files_utils.h"
#include "time_utils.h"

#include <unordered_map>
#include <fstream>
#include <iostream>
#include <cassert>

using std::cout;
using std::endl;


nlohmann::json read_json(const std::string &path_to_json) {
    std::ifstream read_stream(path_to_json);
    nlohmann::json result;
    result = nlohmann::json::parse(read_stream);
    read_stream.close();
    return result;
}

void save_json(const nlohmann::json &json_to_save, const std::string &filename) {
    std::ofstream file(filename);
    file << json_to_save;
}

ProblemObjects read_problem_objects(const std::string &path_to_request) {
    std::unordered_map<std::string, Location> locations;
    std::unordered_map<std::string, Courier> couriers;
    std::unordered_map<std::string, Depot> depots;

    nlohmann::json request_json = read_json(path_to_request);

    for (const auto &location_json: request_json["locations"]) {
        const auto time_window_parsed = time_window_to_begin_seconds_end_seconds(location_json["time_window"]);
        locations.insert({
                                 location_json["id"],
                                 location_json.get<Location>()
                         });
    }

    for (const auto &courier_json: request_json["vehicles"]) {
        couriers.insert({
                                courier_json["id"],
                                courier_json.get<Courier>()
                        });
    }

    for (const auto &depot_json: request_json["depots"]) {
        depots.insert({
                              depot_json["id"],
                              depot_json.get<Depot>()
                      });
    }

    return {locations, couriers, depots};
}

Penalties read_penalties(const std::string &path_to_request) {
    nlohmann::json request_json = read_json(path_to_request);

    float distance_penalty_multiplier = request_json["vehicles"][0]["cost"]["km"];
    for (const auto &vehicle: request_json["vehicles"]) {
        assert(vehicle["cost"]["km"] == distance_penalty_multiplier);
        assert(vehicle["cost"]["hour"] == 0);
        assert(vehicle["cost"]["fixed"] == 0);
    }

    float global_proximity_factor = 0;
    if (request_json["options"].contains("global_proximity_factor")) {
        global_proximity_factor = request_json["options"]["global_proximity_factor"];
    }

    float out_of_time_penalty_per_minute = request_json["locations"][0]["penalty"]["out_of_time"]["minute"];
    for (const auto& location: request_json["locations"]) {
        assert(location["penalty"]["out_of_time"]["fixed"] == 0);
        assert(location["penalty"]["out_of_time"]["minute"] == out_of_time_penalty_per_minute);
    }

    return {
            distance_penalty_multiplier,
            global_proximity_factor,
            out_of_time_penalty_per_minute
    };
}

ProblemDescription read_request_and_get_euclidean_problem(const std::string &path_to_request) {
    nlohmann::json request_json = read_json(path_to_request);
    assert(request_json["options"]["matrix_router"] == "geodesic");
    assert(!request_json["options"].contains("routing_mode") || request_json["options"]["routing_mode"] == "driving");
    for (const auto &vehicle: request_json["vehicles"]) {
        assert(!vehicle.contains("routing_mode") || vehicle["routing_mode"] == "driving");
    }

    ProblemObjects problem_objects = read_problem_objects(path_to_request);
    DistanceMatrix distance_matrix = get_euclidean_distance_matrix(problem_objects);
    TimeMatrix time_matrix = get_geodesic_time_matrix(distance_matrix, "driving");
    Penalties penalties = read_penalties(path_to_request);

    return {
            problem_objects.locations,
            problem_objects.couriers,
            problem_objects.depots,
            distance_matrix,
            time_matrix,
            penalties
    };
}

ProblemDescription read_problem_description(const std::string &path_to_problem_description) {
    nlohmann::json problem_description_json = read_json(path_to_problem_description);
    ProblemDescription result = problem_description_json.get<ProblemDescription>();
    return result;
}
