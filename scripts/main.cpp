#include "../src/algorithms/algorithm.h"
#include "../src/algorithms/annealing/annealing_algorithm.h"
#include "../src/problem/problem_initialization/problem_initialization_simple.h"
#include "../src/utils/files_utils.h"
#include "../src/utils/random_utils.h"

#include <filesystem>
#include <iostream>
#include <fstream>

void set_working_directory_to_project_root() {
    std::string file_path = __FILE__;
    std::string dir_path = file_path.substr(0, file_path.rfind('/'));
    std::filesystem::current_path(dir_path + "/..");
}

int main() {
    fix_random_seed(42);
    set_working_directory_to_project_root();

    std::string test_request_path = "test_data/inputs/simple_test_1/request.json";
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

    algorithm.solve_problem();

    const auto &penalty_history = algorithm.get_penalty_history();
    std::cout << std::endl << "First penalty change: " << penalty_history[0][0] << " -> "
              << penalty_history[0][penalty_history[0].size() - 1];

    algorithm.save_checkpoints("test_data/results/annealing/simple_test_1/checkpoints_1.json");
    algorithm.save_penalty_history("test_data/results/annealing/simple_test_1/penalty_history_1.json");

    // TODO: возможно, стоит заменить вектор из penalty на unordered_map из penalty?
    // TODO наверное нужно как-то разделить file_utils
    // TODO: добавить тесты на всякие утили хотя бы
    // TODO: добавить визуализацию чекпоинтов (обновляющиеся маршруты) и добавить графики для penalty
    // TODO: а что делать с различными депо и хочу ли я вообще несколько депо поддерживать?
    // TODO: добавить больше мутаций
    // TODO: добавить больше penalty? (но это с низким приоритетом задача)
    // TODO: документация к питоновским функциям

    return 0;
}
