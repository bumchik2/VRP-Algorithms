//
// Created by eliseysudakov on 10/2/22.
//

#pragma once

#include <string>

class Depot {
public:
    Depot(std::string id, float lat, float lon) :
            id(std::move(id)), lat(lat), lon(lon) {}

    std::string id;
    float lat = -1.;
    float lon = -1.;
};
