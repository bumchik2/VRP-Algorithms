//
// Created by eliseysudakov on 10/2/22.
//

#pragma once

#include <string>
#include <utility>
#include "../../json/single_include/nlohmann/json.hpp"

class Location {
public:
    Location() = default;  // default constructor for nlohmann::json integrations

    Location(std::string id, std::string depot_id, float lat, float lon,
             int time_window_start_s, int time_window_end_s) :
            id(std::move(id)), depot_id(std::move(depot_id)), lat(lat), lon(lon),
            time_window_start_s(time_window_start_s), time_window_end_s(time_window_end_s) {}

    std::string id;
    std::string depot_id;
    float lat = -1.;
    float lon = -1.;
    int time_window_start_s{};
    int time_window_end_s{};

    [[nodiscard]] std::string get_time_window_str() const;

    friend bool operator == (Location const&, Location const&);  // only used in google tests
};

void to_json(nlohmann::json &j, const Location &location);

void from_json(const nlohmann::json &j, Location &location);

bool operator == (Location const&, Location const&);
