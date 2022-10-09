#include "../src/algorithms/algorithm.h"
#include "../src/algorithms/annealing/annealing_algorithm.h"
#include "../src/problem/problem_initialization/problem_initialization_simple.h"
#include "../src/utils/files_utils.h"
#include "../src/utils/random_utils.h"

#include <iostream>
#include <fstream>

void solve_problem(
        const ProblemDescription &problem_description,
        ProblemSolution &problem_solution,
        const ProblemInitialization &problem_initialization,
        Algorithm &algorithm) {
    algorithm.solve_problem();
    // TODO: penalties is a part of the problem description, not the part of the algorithm.
}


int main() {
    fix_random_seed(42);

    std::string test_data_folder = "test_data/simple_test_data/";
    ProblemDescription problem_description = read_euclidean_problem(test_data_folder);
    ProblemSolution problem_solution(problem_description);
    ProblemInitializationSimple problem_initialization;
    problem_initialization.initialize(problem_description, problem_solution);

    int n_iterations = 100'000;
    float initial_temperature = 10'000;
    AnnealingAlgorithm algorithm(
            problem_description,
            problem_solution,
            n_iterations,
            initial_temperature);

    solve_problem(problem_description,
                  problem_solution,
                  problem_initialization,
                  algorithm);

    const auto& history = algorithm.get_history();
    std::cout << history[0][0] << " -> " << history[0][history[0].size() - 1];

    return 0;
}
