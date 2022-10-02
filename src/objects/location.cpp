//
// Created by eliseysudakov on 10/2/22.
//

#include "location.h"
#include "../utils/time_utils.h"

#include <string>

std::string Location::get_time_window_str() const {
    return seconds_to_datetime_string(static_cast<int>(time_window_start_s)) + "-" +
           seconds_to_datetime_string(static_cast<int>(time_window_end_s));
}