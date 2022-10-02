cmake_minimum_required(VERSION 3.23)
project(VRP_Algorithms)

set(CMAKE_CXX_STANDARD 14)

file(GLOB main_src
        "src/*.h"
        "src/*.cpp"
        )

add_executable(VRP_Algorithms_main
        scripts/main.cpp ${main_src})

add_executable(VRP_Algorithms_test
        scripts/test.cpp)
