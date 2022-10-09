//
// Created by eliseysudakov on 10/2/22.
//

#pragma once

#include <string>

class Courier {
public:
    Courier(std::string id, std::string depot_id) :
            id(std::move(id)), depot_id(std::move(depot_id)) {}

    std::string id;
    std::string depot_id;
};
