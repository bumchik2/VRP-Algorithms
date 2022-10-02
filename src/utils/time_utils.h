//
// Created by eliseysudakov on 10/2/22.
//

#pragma once

#include "../utils/common_utils.h"

#include <string>

std::string seconds_to_datetime_string(int seconds) {
    // date format: YYYY-MM-DD
    // return format: [D.]HH:MM:SS

    int s = seconds % 60;
    int m = seconds / 60 % 60;
    int h = seconds / 3600 % 24;
    int d = seconds / 86400;

    std::string ss = add_zeros_prefix(int_to_string(s), 2);
    std::string mm = add_zeros_prefix(int_to_string(m), 2);
    std::string hh = add_zeros_prefix(int_to_string(h), 2);
    std::string D;
    if (d > 0) {
        D = int_to_string(d) + ".";
    }

    return D + hh + mm + ss;
}

