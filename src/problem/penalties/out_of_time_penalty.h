//
// Created by eliseysudakov on 12/10/22.
//

#pragma once

#include "penalty.h"
#include "../../utils/common_utils.h"

#include <string>

class OutOfTimePenalty : public Penalty {
public:
    explicit OutOfTimePenalty(float penalty_multiplier):
            Penalty(penalty_multiplier, PER_ROUTE_PENALTY) {}

    ~OutOfTimePenalty() override = default;

    [[nodiscard]] float get_penalty(const ProblemDescription &problem_description, const std::vector<Route> &routes) const override;

    [[nodiscard]] std::string get_short_name() const override {
        return "out-of-time-penalty";
    }

    [[nodiscard]] std::string get_name() const override {
        return "[Out of time penalty with multiplier " + float_to_string(_penalty_multiplier, 2) +  " per minute]";
    }
};
