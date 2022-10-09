//
// Created by eliseysudakov on 10/2/22.
//

#pragma once

#include "mutation.h"

#include <string>

class SwapVertexMutation : public Mutation {
public:
    explicit SwapVertexMutation(): Mutation() {}

    std::string get_name() const override {
        return "[Swap Vertex Mutation]";
    }

    void mutate(ProblemSolution& problem_solution) const override;

private:

    std::vector<int> _choose_mutation_parameters_(const ProblemSolution& problem_solution) const;
    std::vector<int> _get_modified_routes_indices(const ProblemSolution &problem_solution) const override;
};
