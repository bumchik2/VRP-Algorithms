//
// Created by eliseysudakov on 10/2/22.
//

#include "common_utils.h"

#include <sstream>
#include <iomanip>

std::string float_to_string(float x, int decimal_digits) {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(decimal_digits) << x;
    return stream.str();
}

std::string int_to_string(int n) {
    return std::to_string(n);
}

int string_to_int(const std::string& n_str) {
    return std::stoi(n_str);
}

std::string add_zeros_prefix(std::string s, int min_length) {
    s.insert(0, min_length - s.size(), '0');
    return s;
}
