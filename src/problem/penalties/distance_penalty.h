//
// Created by eliseysudakov on 10/2/22.
//

#pragma once

#include "penalty.h"
#include "../../utils/common_utils.h"

#include <string>

class DistancePenalty : public Penalty {
public:
    explicit DistancePenalty(float penalty_multiplier) :
            Penalty(penalty_multiplier, PER_ROUTE_PENALTY) {}

    ~DistancePenalty() override = default;

    [[nodiscard]] float get_penalty(const ProblemDescription &problem_description, const std::vector<Route> &routes) const override;

    [[nodiscard]] std::string get_short_name() const override {
        return "distance-penalty";
    }

    [[nodiscard]] std::string get_name() const override {
        return "[Distance penalty with multiplier " + float_to_string(_penalty_multiplier, 2) + "]";
    }
};
