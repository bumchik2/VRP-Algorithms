//
// Created by eliseysudakov on 10/2/22.
//

#pragma once

#include <string>
#include <utility>

class Location {
public:
    Location(std::string id, std::string depot_id, float lat, float lon, float time_window_start_s, float time_window_end_s):
            id(std::move(id)), depot_id(std::move(depot_id)), lat(lat), lon(lon), time_window_start_s(time_window_start_s), time_window_end_s(time_window_end_s) {}

    std::string id;
    std::string depot_id;
    float lat = -1.;
    float lon = -1.;
    float time_window_start_s;
    float time_window_end_s;

    std::string get_time_window_str() const;
};
