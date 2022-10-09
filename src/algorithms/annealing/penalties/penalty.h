//
// Created by eliseysudakov on 10/2/22.
//

#pragma once

#include "../../../objects/route.h"
#include "../../../problem/problem_description.h"

#include <vector>
#include <string>

enum PenaltyType {
    // Penalty is evaluated for each route independently
    PER_ROUTE_PENALTY,

    // Penalty is evaluated for the problem as a whole
    PER_PROBLEM_PENALTY,
};

class Penalty {
public:
    Penalty(float penalty_multiplier, PenaltyType penalty_type):
            _penalty_multiplier(penalty_multiplier), penalty_type(penalty_type) {}

    virtual ~Penalty() = default;

    virtual float get_penalty(const ProblemDescription& problem_description, const std::vector<Route>& routes) const {
        throw std::runtime_error("Calling get_penalty from base Penalty class");
    };

    virtual std::string get_name() const {
        throw std::runtime_error("Calling get_name from base Penalty class");
    }

    PenaltyType penalty_type;

protected:
    float _penalty_multiplier;
};
