//
// Created by eliseysudakov on 10/2/22.
//

#pragma once

#include <string>

std::string float_to_string(float x, int decimal_digits);
std::string int_to_string(int n);
int string_to_int(const std::string& n_str);

std::string add_zeros_prefix(std::string s, int min_length);
