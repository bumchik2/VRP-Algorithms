//
// Created by eliseysudakov on 10/2/22.
//

#pragma once

#include <vector>
#include <string>

class Route {
public:
    std::vector<std::string> location_ids;
    std::string vehicle_id;

    bool empty() const {
        return location_ids.empty();
    }
};
