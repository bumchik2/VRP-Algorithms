//
// Created by eliseysudakov on 10/2/22.
//

#pragma once

#include <string>
#include <vector>
#include <memory>

#include "../penalties/penalty.h"
#include "../../../problem/problem_solution.h"

class Mutation {
public:
    Mutation(int random_seed): _random_seed(random_seed) {}

    virtual std::string get_name() const {}

    virtual void mutate(ProblemSolution& problem_solution) const {};
    virtual float get_delta_penalty(ProblemSolution &problem_solution,
            const std::vector<std::shared_ptr<Penalty>> &penalties) const;

private:
    int _random_seed;

    virtual std::vector<int> _get_modified_routes_indices(const ProblemSolution& problem_solution) const {};

    static float _calculate_penalty_part(const ProblemSolution &problem_solution,
                                         const std::vector<std::shared_ptr<Penalty>> &penalties,
                                         const std::vector<int> &modified_routes_indices);

protected:
    void _fix_random_seed() const;
};
