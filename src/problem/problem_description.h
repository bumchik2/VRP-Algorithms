//
// Created by eliseysudakov on 10/2/22.
//

#pragma once

#include "../matrices/time_matrix.h"
#include "../matrices/distance_matrix.h"
#include "../objects/location.h"
#include "../objects/courier.h"
#include "../objects/depot.h"

#include <unordered_map>

class ProblemDescription {
public:
    ProblemDescription(std::unordered_map<std::string, Location> locations,
            std::unordered_map<std::string, Courier> couriers,
            std::unordered_map<std::string, Depot> depots,
            DistanceMatrix distance_matrix, TimeMatrix time_matrix):
            locations(std::move(locations)), couriers(std::move(couriers)), depots(std::move(depots)),
            distance_matrix(std::move(distance_matrix)), time_matrix(std::move(time_matrix)) {}

    const std::unordered_map<std::string, Location> locations;
    const std::unordered_map<std::string, Courier> couriers;
    const std::unordered_map<std::string, Depot> depots;

    const DistanceMatrix distance_matrix;
    const TimeMatrix time_matrix;
};
