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
    const std::unordered_map<std::string, Location> locations;
    const std::unordered_map<std::string, Courier> couriers;
    const std::unordered_map<std::string, Depot> depots;

    const TimeMatrix time_matrix;
    const DistanceMatrix distance_matrix;
};
