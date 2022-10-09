cmake_minimum_required(VERSION 3.23)
project(VRP_Algorithms)

set(CMAKE_CXX_STANDARD 14)

# not working for some reason
#file(GLOB SOURCES
#        CONFIGURE_DEPENDS
#        "/home/eliseysudakov/VRP-Algorithms/src/*.cpp"
#        "/home/eliseysudakov/VRP-Algorithms/src/*.h"
#        )

set(SOURCES
        json/single_include/nlohmann/json.hpp
        src/algorithms/annealing/penalties/distance_penalty.cpp
        src/algorithms/annealing/penalties/penalty.cpp
        src/algorithms/annealing/mutations/mutation.cpp
        src/algorithms/annealing/mutations/swap_vertex_mutation.cpp
        src/algorithms/annealing/annealing_algorithm.cpp
        src/algorithms/algorithm.cpp
        src/objects/courier.cpp
        src/objects/location.cpp
        src/objects/depot.cpp
        src/matrices/distance_matrix.cpp
        src/matrices/time_matrix.cpp
        src/problem/problem_initialization/problem_initialization.cpp
        src/problem/problem_initialization/problem_initialization_simple.cpp
        src/problem/problem_description.cpp
        src/problem/problem_solution.cpp
        src/utils/common_utils.cpp
        src/utils/file_utils.cpp
        src/utils/random_utils.cpp
        src/utils/time_utils.cpp)

add_executable(VRP_Algorithms_main
        scripts/main.cpp json/single_include/nlohmann/json.hpp ${SOURCES})

add_executable(VRP_Algorithms_test
        scripts/test.cpp)
