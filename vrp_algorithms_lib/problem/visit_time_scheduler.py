from typing import List, Union

from vrp_algorithms_lib.problem.models import ProblemDescription, Route, DepotId, LocationId


class VisitTimeScheduler:
    @staticmethod
    def get_locations_visit_times(problem_description: ProblemDescription, route: Route) -> List[int]:
        if len(route.location_ids) == 0:
            return []

        current_time_s = 0
        previous_id: Union[DepotId, LocationId] = problem_description.locations[route.location_ids[0]].depot_id
        visit_times = []

        for i, location_id in enumerate(route.location_ids):
            location = problem_description.locations[location_id]

            if i == 0:
                travel_time_hours = problem_description.time_matrix.depots_to_locations_travel_times[
                    previous_id][location_id]
            else:
                travel_time_hours = problem_description.time_matrix.locations_to_locations_travel_times[
                    previous_id][location_id]
            travel_time_s = travel_time_hours * 3600.0
            current_time_s += int(travel_time_s)

            current_time_s = max(current_time_s, location.time_window_start_s)

            visit_times.append(current_time_s)
            previous_id = location_id

        return visit_times
