#include <iostream>

#include "../src/algorithms/algorithm.h"
#include "../src/algorithms/annealing/annealing_algorithm.h"
#include "../src/problem/problem_initialization/problem_initialization_simple.h"


void solve_problem(
        const ProblemDescription& problem_description,
        ProblemSolution problem_solution,
        const ProblemInitialization& problem_initialization,
        Algorithm& algorithm) {
    problem_initialization.initialize(problem_description, problem_solution);
    algorithm.solve_problem();
    // TODO: penalties is a part of the problem description, not the part of the algorithm.
}


int main() {
    ProblemDescription problem_description;
    ProblemSolution problem_solution(problem_description);
    ProblemInitializationSimple problem_initialization;

    int n_iterations = 100'000;
    float initial_temperature = 10'000;
    float distance_penalty_multiplier = 10.;
    AnnealingAlgorithm algorithm(
            problem_description,
            problem_solution,
            n_iterations,
            initial_temperature,
            distance_penalty_multiplier);
    return 0;
}
