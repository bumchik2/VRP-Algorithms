//
// Created by eliseysudakov on 10/2/22.
//

#pragma once

#include <string>

struct DatetimeObject {
    int days;
    int hours;
    int minutes;
    int seconds;

    [[nodiscard]] std::string to_string() const;
};

std::string seconds_to_datetime_string(int seconds);

int datetime_string_to_seconds(const std::string &datetime_string);

DatetimeObject parse_datetime_string(const std::string &datetime_string);

std::pair<int, int> time_window_to_begin_seconds_end_seconds(const std::string &time_window);
