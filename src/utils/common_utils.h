//
// Created by eliseysudakov on 10/2/22.
//

#pragma once

#include <string>

std::string float_to_string(float x, int decimal_digits);
std::string int_to_string(int n);
int string_to_int(const std::string& n_str);

std::string add_zeros_prefix(std::string s, int min_length);

int char_to_digit(char c);

int parse_int_from_string(const std::string& s, int start_pos, int end_pos);
