//
// Created by eliseysudakov on 12/10/22.
//

#include "out_of_time_penalty.h"
#include "../visit_time_scheduler.h"

float OutOfTimePenalty::get_penalty(
        const ProblemDescription &problem_description, const std::vector<Route> &routes) const {
    float penalty = 0;

    for (const auto &route: routes) {
        if (route.location_ids.empty()) {
            continue;
        }

        std::vector<int> visit_times = VisitTimeScheduler::get_locations_visit_times(problem_description, route);
        for (int i = 0; i < static_cast<int>(visit_times.size()); ++i) {
            const std::string &location_id = route.location_ids[i];
            const Location &location = problem_description.locations.at(location_id);
            int visit_time = visit_times[i];

            int out_of_time_s = std::max(0, std::max(visit_time - location.time_window_end_s,
                                                     location.time_window_start_s - visit_time));
            float out_of_time_minutes = static_cast<float>(out_of_time_s) / 60.0f;
            penalty += out_of_time_minutes * _penalty_multiplier;
        }
    }

    return penalty;
}
