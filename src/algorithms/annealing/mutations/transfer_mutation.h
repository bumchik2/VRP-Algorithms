//
// Created by eliseysudakov on 10/16/22.
//

#pragma once

#pragma once

#include "mutation.h"

#include <string>

class TransferMutation : public Mutation {
public:
    explicit TransferMutation(): Mutation() {}

    [[nodiscard]] std::string get_name() const override {
        return "[Swap Vertex Mutation]";
    }

    void mutate(ProblemSolution& problem_solution) const override;

private:

    [[nodiscard]] std::vector<int> _choose_mutation_parameters_(const ProblemSolution& problem_solution) const;
    [[nodiscard]] std::vector<int> _get_modified_routes_indices(const ProblemSolution &problem_solution) const override;
};
