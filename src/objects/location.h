//
// Created by eliseysudakov on 10/2/22.
//

#pragma once

#include <string>

class Location {
public:
    std::string id;
    float lat = -1.;
    float lon = -1.;
    float time_window_start_s;
    float time_window_end_s;

    std::string get_time_window_str() const;
};
