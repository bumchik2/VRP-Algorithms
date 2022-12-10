//
// Created by eliseysudakov on 12/10/22.
//

#include "visit_time_scheduler.h"

std::vector<int> VisitTimeScheduler::get_locations_visit_times(
        const ProblemDescription& problem_description, const Route& route) {
    if (route.empty()) {
        return {};
    }

    std::string previous_id = problem_description.locations.at(route.location_ids[0]).depot_id;
    int current_time_s = 0;
    std::vector<int> result;

    for (int i = 0; i < static_cast<int>(route.location_ids.size()); ++i) {
        const std::string& location_id = route.location_ids[i];
        const Location& location = problem_description.locations.at(location_id);

        float travel_time_hours;
        if (i == 0) {
            travel_time_hours = problem_description.time_matrix.get_travel_time_depot_to_location(previous_id, location_id);
        } else {
            travel_time_hours = problem_description.time_matrix.get_travel_time_location_to_location(previous_id, location_id);
        }

        int travel_time_seconds = static_cast<int>(travel_time_hours * 3600.0);
        current_time_s += travel_time_seconds;
        current_time_s = std::max(location.time_window_start_s, current_time_s);

        result.push_back(current_time_s);
        previous_id = location_id;
    }

    return result;
}
