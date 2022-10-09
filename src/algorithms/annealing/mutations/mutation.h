//
// Created by eliseysudakov on 10/2/22.
//

#pragma once

#include <string>
#include <vector>
#include <memory>

#include "../../../problem/penalties/penalty.h"
#include "../../../problem/problem_solution.h"

class Mutation {
public:
    explicit Mutation() = default;

    void set_random_seed(int random_seed);

    virtual std::string get_name() const = 0;

    virtual void mutate(ProblemSolution &problem_solution) const {};

    std::vector<float> get_delta_penalties(ProblemSolution &problem_solution,
                                           const std::vector<std::shared_ptr<Penalty>> &penalties) const;

private:
    int _random_seed = -1;

    virtual std::vector<int> _get_modified_routes_indices(const ProblemSolution &problem_solution) const = 0;

    static std::vector<float> _calculate_penalty_part(const ProblemSolution &problem_solution,
                                                      const std::vector<std::shared_ptr<Penalty>> &penalties,
                                                      const std::vector<int> &modified_routes_indices);

protected:
    void _fix_random_seed() const;
};
