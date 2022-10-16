//
// Created by eliseysudakov on 10/2/22.
//

#include "common_utils.h"
#include "time_utils.h"

#include <utility>

std::string seconds_to_datetime_string(int seconds) {
    // return format: [D.]HH:MM:SS

    int s = seconds % 60;
    int m = seconds / 60 % 60;
    int h = seconds / 3600 % 24;
    int d = seconds / 86400;

    DatetimeObject datetime_object {
        d, h, m, s
    };

    return datetime_object.to_string();
}

std::string DatetimeObject::to_string() const {
    std::string ss = add_zeros_prefix(int_to_string(seconds), 2);
    std::string mm = add_zeros_prefix(int_to_string(minutes), 2);
    std::string hh = add_zeros_prefix(int_to_string(hours), 2);

    std::string D;
    if (days > 0) {
        D = int_to_string(days) + ".";
    }

    return D + hh + ":" + mm + ":" + ss;
}

DatetimeObject parse_datetime_string(const std::string& datetime_string) {
    // input format: [D.]HH:MM:SS
    int days = 0;
    int hours_start_pos = 0;
    if (datetime_string.find('.') != std::string::npos) {
        hours_start_pos = static_cast<int>(datetime_string.find('.')) + 1;
        days = parse_int_from_string(datetime_string,0, static_cast<int>(datetime_string.find('.')));
    }
    int hours_end_pos = hours_start_pos + 2;
    int minutes_start_pos = hours_end_pos + 1;
    int minutes_end_pos = minutes_start_pos + 2;
    int seconds_start_pos = minutes_end_pos + 1;
    int seconds_end_pos = seconds_start_pos + 2;

    int hours = parse_int_from_string(datetime_string, hours_start_pos, hours_end_pos);
    int minutes = parse_int_from_string(datetime_string, minutes_start_pos, minutes_end_pos);
    int seconds = parse_int_from_string(datetime_string, seconds_start_pos, seconds_end_pos);

    return {
        days,
        hours,
        minutes,
        seconds
    };
}

std::pair<float, float> time_window_to_begin_seconds_end_seconds(const std::string &time_window) {
    int separator_pos = static_cast<int>(time_window.find('-'));
    std::string time_window_start = time_window.substr(0, separator_pos);
    std::string time_window_end = time_window.substr(separator_pos + 1, time_window.size() - separator_pos - 1);
    return {datetime_string_to_seconds(time_window_start), datetime_string_to_seconds(time_window_end)};
}

int datetime_string_to_seconds(const std::string& datetime_string) {
    // input format: [D.]HH:MM:SS
    DatetimeObject datetime_object = parse_datetime_string(datetime_string);
    return datetime_object.days * 86400 + datetime_object.hours * 3600 + datetime_object.minutes * 60 + datetime_object.seconds;
}
