#include <iostream>
#include "../src/utils/time_utils.h"
#include "../src/utils/common_utils.h"
#include "gtest/gtest.h"
using std::cout;
using std::endl;

TEST(TimeUtilsTests, seconds_to_datetime_string_test) {
    ASSERT_EQ(seconds_to_datetime_string(12345), "03:25:45");
}

TEST(TimeUtilsTests, datetime_string_to_seconds_test) {
    ASSERT_EQ(datetime_string_to_seconds("03:25:45"), 12345);
}

TEST(TimeUtilsTests, time_window_to_begin_seconds_end_seconds_test) {
    const auto res = time_window_to_begin_seconds_end_seconds("02:15:47-2.14:50:00");
    ASSERT_EQ(res.first, 8147);
    ASSERT_EQ(res.second, 226200);
}

TEST(CommonUtilsTests, float_to_string_test) {
    ASSERT_EQ(float_to_string(3.1415, 3), "3.141");
    ASSERT_EQ(float_to_string(3., 2), "3.00");
}

TEST(CommonUtilsTests, int_to_string_test) {
    ASSERT_EQ(int_to_string(123), "123");
}

TEST(CommonUtilsTests, string_to_int_test) {
    ASSERT_EQ(string_to_int("123"), 123);
}

TEST(CommonUtilsTests, add_zeros_prefix_test) {
    ASSERT_EQ(add_zeros_prefix("123", 3), "123");
    ASSERT_EQ(add_zeros_prefix("123", 5), "00123");
}

TEST(CommonUtilsTests, char_to_digit_test) {
    ASSERT_EQ(char_to_digit('1'), 1);
}

TEST(CommonUtilsTests, parse_int_from_string_test) {
    ASSERT_EQ(parse_int_from_string("aa100b", 2, 4), 10);
}
