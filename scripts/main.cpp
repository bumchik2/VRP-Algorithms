#include "../src/algorithms/algorithm.h"
#include "../src/algorithms/annealing/annealing_algorithm.h"
#include "../src/problem/problem_initialization/problem_initialization_simple.h"
#include "../src/utils/files_utils.h"
#include "../src/utils/random_utils.h"

#include <filesystem>
#include <iostream>
#include <fstream>

void solve_problem(
        const ProblemDescription &problem_description,
        ProblemSolution &problem_solution,
        const ProblemInitialization &blem_initialization,
        Algorithm &algorithm) {
    algorithm.solve_problem();
}

void set_working_directory_to_project_root() {
    std::string file_path = __FILE__;
    std::string dir_path = file_path.substr(0, file_path.rfind('/'));
    std::filesystem::current_path(dir_path + "/..");
}

int main() {
    fix_random_seed(42);
    set_working_directory_to_project_root();

    std::string test_request_path = "test_data/simple_test_data/simple_request.json";
    ProblemDescription problem_description = read_euclidean_problem(test_request_path);
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
    std::cout << std::endl << "First penalty change: " << history[0][0] << " -> " << history[0][history[0].size() - 1];

    // TODO: добавить тесты на всякие утили хотя бы
    // TODO: save history and checkpoints to some files
    // TODO: добавить визуализацию чекпоинтов (обновляющиеся маршруты) и добавить графики для penalty
    // TODO: а что делать с различными депо и хочу ли я вообще несколько депо поддерживать?

    return 0;
}
