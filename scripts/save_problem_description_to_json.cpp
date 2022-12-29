//
// Created by eliseysudakov on 10/29/22.
//

#include "../src/algorithms/algorithm.h"
#include "../src/utils/files_utils.h"
#include "../src/utils/random_utils.h"

#include <iostream>


int main(int argc, char *argv[]) {
    // usage: ./main.exe path_to_request path_to_problem_description

    std::cout << "You have entered " << argc << " arguments:" << "\n";
    for (int i = 0; i < argc; ++i) {
        std::cout << argv[i] << "\n";
    }

    const std::string path_to_request = argv[1];
    const std::string path_to_problem_description = argv[2];

    ProblemDescription problem_description = read_request_and_get_euclidean_problem(path_to_request);
    std::cout << "Read request and got euclidean problem successfully" << std::endl;
    save_problem_description_to_json(problem_description, path_to_problem_description);
}
