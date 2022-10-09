cmake_minimum_required(VERSION 3.16.3)
project(VRP_Algorithms)

set(CMAKE_CXX_STANDARD 17)

set(SOURCES
        src/problem/penalties/distance_penalty.cpp
        src/problem/penalties/penalty.cpp
        src/algorithms/annealing/mutations/mutation.cpp
        src/algorithms/annealing/mutations/swap_vertex_mutation.cpp
        src/algorithms/annealing/annealing_algorithm.cpp
        src/algorithms/iterative_algorithm.cpp
        src/algorithms/algorithm.cpp
        src/objects/courier.cpp
        src/objects/location.cpp
        src/objects/depot.cpp
        src/objects/route.cpp
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
        scripts/main.cpp json/single_include/nlohmann/json.hpp tqdm/tqdm.h ${SOURCES})

add_executable(VRP_Algorithms_test
        scripts/test.cpp json/single_include/nlohmann/json.hpp tqdm/tqdm.h ${SOURCES})

add_executable(VRP_Algorithms_test_includes
        scripts/test_includes.cpp)
