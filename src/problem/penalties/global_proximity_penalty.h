//
// Created by eliseysudakov on 11/28/22.
//

#pragma once

#include "penalty.h"
#include "../../utils/common_utils.h"

#include <string>

class GlobalProximityPenalty : public Penalty {
public:
    explicit GlobalProximityPenalty(float penalty_multiplier) :
            Penalty(penalty_multiplier, PER_ROUTE_PENALTY) {}

    ~GlobalProximityPenalty() override = default;

    [[nodiscard]] float get_penalty(const ProblemDescription &problem_description, const std::vector<Route> &routes) const override;

    [[nodiscard]] std::string get_short_name() const override {
        return "global-proximity-penalty";
    }

    [[nodiscard]] std::string get_name() const override {
        return "[Global proximity penalty with multiplier " + float_to_string(_penalty_multiplier, 2) + "]";
    }
};
