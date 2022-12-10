//
// Created by eliseysudakov on 12/10/22.
//

#pragma once

#include <vector>
#include "problem_description.h"
#include "../objects/route.h"

class VisitTimeScheduler {
public:
    static std::vector<int> get_locations_visit_times(const ProblemDescription& problem_description, const Route& route);
};
