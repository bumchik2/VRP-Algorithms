cmake_minimum_required(VERSION 3.16.3)
project(VRP_Algorithms)

set(CMAKE_CXX_STANDARD 17)

set(SOURCES
        src/problem/penalties/distance_penalty.cpp
        src/problem/penalties/penalty.cpp
        src/algorithms/annealing/mutations/mutation.cpp
        src/algorithms/annealing/mutations/swap_vertex_mutation.cpp
        src/algorithms/annealing/mutations/transfer_mutation.cpp
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

add_subdirectory(google_tests)

include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

add_executable(run_google_tests scripts/test.cpp ${SOURCES})
target_link_libraries(run_google_tests gtest gtest_main)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

add_executable(VRP_Algorithms_greedy_algorithm_main
        scripts/greedy_algorithm_main.cpp json/single_include/nlohmann/json.hpp tqdm/tqdm.h ${SOURCES})

target_link_libraries(VRP_Algorithms_greedy_algorithm_main ${PYTHON_LIBRARIES})

add_executable(VRP_Algorithms_annealing_algorithm_main
        scripts/annealing_algorithm_main.cpp json/single_include/nlohmann/json.hpp tqdm/tqdm.h ${SOURCES})

add_executable(VRP_Algorithms_save_problem_description_to_json
        scripts/save_problem_description_to_json.cpp json/single_include/nlohmann/json.hpp ${SOURCES})
