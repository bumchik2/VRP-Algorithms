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
        const ProblemInitialization &blem_initialization,
        Algorithm &algorithm) {
    algorithm.solve_problem();
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

    const auto &history = algorithm.get_history();
    std::cout << "First penalty change: " << history[0][0] << " -> " << history[0][history[0].size() - 1];

    // TODO: refactor reading files and getting eucledian problems
    // TODO: save history and checkpoints somewhere
    // TODO: очень хочется как-то поженить форматы настоящего солвера и моего, чтобы можно было и тут и там одинаковые запросы отправлять
    // TODO: а что делать с различными депо и хочу ли я вообще несколько депо поддерживать?

    return 0;
}
