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

void set_working_directory_to_project_root() {
    std::string file_path = __FILE__;
    std::string dir_path = file_path.substr(0, file_path.rfind('/'));
    std::filesystem::current_path(dir_path + "/..");
}

int main(int argc, char *argv[]) {
    // usage: ./main.exe <test_name> <n_iterations> <initial_temperature>

    std::cout << "You have entered " << argc << " arguments:" << "\n";
    for (int i = 0; i < argc; ++i) {
        std::cout << argv[i] << "\n";
    }

    fix_random_seed(42);
    set_working_directory_to_project_root();

    const std::string test_name = argv[1];
    int n_iterations = string_to_int(argv[2]);
    float initial_temperature = static_cast<float>(string_to_int(argv[3]));

    std::string test_request_path = "test_data/inputs/" + test_name + "/request.json";
    ProblemDescription problem_description = read_euclidean_problem(test_request_path);
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

    const std::string results_folder = "test_data/results/annealing/" + test_name + "/";
    algorithm.save_routes(results_folder + int_to_string(n_iterations) + "_iterations_routes.json");
    algorithm.save_penalty(results_folder + int_to_string(n_iterations) + "_iterations_penalty.json");
    algorithm.save_checkpoints(results_folder + int_to_string(n_iterations) + "_iterations_checkpoints.json");
    algorithm.save_penalty_history(results_folder + int_to_string(n_iterations) + "_iterations_penalty_history.json");

    // TODO: читать в этом скрипте нужно не request, а problem_desciprition
    // TODO: добавить penalty за балансировку маршрутов
    // TODO: нужно написать питоновский класс / функцию для вычисления Penalty (и соответствующую pydantic модель добавить)

    // TODO: перевести viz на pydantic модели (пониженный приоритет)

    // TODO наверное нужно как-то разделить file_utils (низкий приоритет)
    // TODO: документация к питоновским функциям (низкий приоритет)
    // TODO: добавить больше мутаций (низкий приоритет)

    return 0;
}
