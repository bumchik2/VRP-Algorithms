//
// Created by eliseysudakov on 10/9/22.
//

#pragma once

#include <string>
#include <unordered_map>

#include "location.h"
#include "courier.h"
#include "depot.h"

struct ProblemObjects {
    std::unordered_map<std::string, Location> locations;
    std::unordered_map<std::string, Courier> couriers;
    std::unordered_map<std::string, Depot> depots;
};
