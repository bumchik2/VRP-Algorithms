//
// Created by eliseysudakov on 10/2/22.
//

#include "location.h"
#include "../utils/time_utils.h"

#include <string>
#include <iostream>

std::string Location::get_time_window_str() const {
    return seconds_to_datetime_string(static_cast<int>(time_window_start_s)) + "-" +
           seconds_to_datetime_string(static_cast<int>(time_window_end_s));
}

void to_json(nlohmann::json &j, const Location &location) {
    j = {
            {"id",                  location.id},
            {"depot_id",            location.depot_id},
            {"point",               {
                                            {"lat", location.lat},
                                            {"lon", location.lon},
                                    }},
            {"time_window_start_s", location.time_window_start_s},
            {"time_window_end_s",   location.time_window_end_s},
    };
}

void from_json(const nlohmann::json &j, Location &location) {
    if (j.contains("time_window")) {  // in yandex solver request time_window is stored for each vehicle
        const std::string time_window_str = j.at("time_window").get<std::string>();
        const auto time_window_pair = time_window_to_begin_seconds_end_seconds(time_window_str);
        location.time_window_start_s = time_window_pair.first;
        location.time_window_end_s = time_window_pair.second;
    } else {  // but in `problem description` time_window_start_s and time_window_end_s are stored
        j.at("time_window_start_s").get_to(location.time_window_start_s);
        j.at("time_window_end_s").get_to(location.time_window_end_s);
    }

    j.at("id").get_to(location.id);
    j.at("depot_id").get_to(location.depot_id);
    j.at("point").at("lat").get_to(location.lat);
    j.at("point").at("lon").get_to(location.lon);
}

bool operator==(Location const &location_1, Location const &location_2) {
    // TODO: why can't this be generated automatically?
    if (location_1.id != location_2.id) {
        return false;
    }
    if (location_1.depot_id != location_2.depot_id) {
        return false;
    }
    if (location_1.lat != location_2.lat) {
        return false;
    }
    if (location_1.lon != location_2.lon) {
        return false;
    }
    if (location_1.time_window_start_s != location_2.time_window_start_s) {
        return false;
    }
    if (location_1.time_window_end_s != location_2.time_window_end_s) {
        return false;
    }
    return true;
}
