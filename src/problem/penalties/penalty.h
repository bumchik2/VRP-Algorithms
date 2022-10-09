//
// Created by eliseysudakov on 10/2/22.
//

#pragma once

#include "../../objects/route.h"

#include <stdexcept>
#include <vector>
#include <string>

class ProblemDescription;

enum PenaltyType {
    // Penalty is evaluated for each route independently
    PER_ROUTE_PENALTY,

    // Penalty is evaluated for the problem as a whole
    PER_PROBLEM_PENALTY,
};

class Penalty {
public:
    Penalty(const ProblemDescription &problem_description, float penalty_multiplier, PenaltyType penalty_type) :
            _penalty_multiplier(penalty_multiplier), penalty_type(penalty_type),
            _problem_description(problem_description) {}

    virtual ~Penalty() = default;

    virtual float get_penalty(const std::vector<Route> &routes) const {
        throw std::runtime_error("Calling get_penalty from base Penalty class");
    };

    virtual std::string get_name() const {
        throw std::runtime_error("Calling get_name from base Penalty class");
    }

    PenaltyType penalty_type;

protected:
    float _penalty_multiplier;
    const ProblemDescription &_problem_description;
};
