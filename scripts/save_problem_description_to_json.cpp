//
// Created by eliseysudakov on 10/29/22.
//

#include "../src/algorithms/algorithm.h"
#include "../src/utils/files_utils.h"
#include "../src/utils/random_utils.h"

#include <filesystem>

void set_working_directory_to_project_root() {
    std::string file_path = __FILE__;
    std::string dir_path = file_path.substr(0, file_path.rfind('/'));
    std::filesystem::current_path(dir_path + "/..");
}

int main() {
    fix_random_seed(42);
    set_working_directory_to_project_root();

    const std::string test_name = "medium_test_1";
    const std::string inputs_folder = "test_data/inputs/";
    std::string test_request_path = inputs_folder + test_name + "/request.json";
    ProblemDescription problem_description = read_euclidean_problem(test_request_path);
    save_problem_description_to_json(problem_description, inputs_folder + test_name + "/problem_description.json");
}
