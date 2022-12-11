//
// Created by eliseysudakov on 12/10/22.
//

#include "../src/algorithms/algorithm.h"
#include "../src/algorithms/iterative_algorithm.h"
#include "../src/algorithms/annealing/annealing_algorithm.h"
#include "../src/problem/problem_initialization/problem_initialization_simple.h"
#include "../src/utils/files_utils.h"
#include "../src/utils/random_utils.h"
#include "../src/utils/common_utils.h"

#include <filesystem>
#include <iostream>
#include <fstream>

int main(int argc, char *argv[]) {
    // usage: ./main.exe <problem_description_path> <routes_path> <n_iterations> <initial_temperature>
    // problem_description_path is the path to problem description json file
    // routes_path is the path where the final routes should be saved
    // n_iterations is the number of iterations of annealing to execute
    // initial_temperature is the initial annealing algorithm temperature

    std::cout << "You have entered " << argc << " arguments:" << "\n";
    for (int i = 0; i < argc; ++i) {
        std::cout << argv[i] << "\n";
    }

    const std::string problem_description_path = argv[1];
    const std::string routes_path = argv[2];
    const int n_iterations = string_to_int(argv[3]);
    const float initial_temperature = static_cast<float>(string_to_int(argv[4]));

    fix_random_seed(42);

    ProblemDescription problem_description = read_problem_description(problem_description_path);
    ProblemSolution problem_solution(problem_description);
    ProblemInitializationSimple problem_initialization;
    problem_initialization.initialize(problem_description, problem_solution);

    AnnealingAlgorithm algorithm(
            problem_description,
            problem_solution,
            n_iterations,
            initial_temperature
    );

    algorithm.solve_problem();
    print_penalty_changes(problem_description, algorithm);
    algorithm.save_routes(routes_path);
    return 0;
}
