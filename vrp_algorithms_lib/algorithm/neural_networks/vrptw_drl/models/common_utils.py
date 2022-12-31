from vrp_algorithms_lib.algorithm.neural_networks.vrptw_drl.objects import ProblemState, Routes, CourierId, LocationId


def choose_next_courier_id(
        problem_state: ProblemState,
        routes: Routes
) -> CourierId:
    min_distance = 10 ** 9
    next_courier_id = None
    for potential_next_courier_idx in range(len(problem_state.problem_description.couriers)):
        potential_next_courier_id = problem_state.idx_to_courier_id[potential_next_courier_idx]
        current_courier_vehicle_state = problem_state.get_vehicle_state_by_courier_id(potential_next_courier_id)
        current_courier_route = routes.get_route_by_vehicle_id(potential_next_courier_id)
        current_filtered_partial_route = current_courier_vehicle_state.get_filtered_partial_route()

        if len(current_filtered_partial_route) == len(current_courier_route.location_ids):
            continue

        potential_next_location_id = current_courier_route.location_ids[len(current_filtered_partial_route)]
        depot_id = problem_state.problem_description.get_depot().id
        if len(current_filtered_partial_route) == 0:
            current_distance = problem_state.problem_description.distance_matrix.depots_to_locations_distances[
                depot_id][potential_next_location_id]
        else:
            current_distance = problem_state.problem_description.distance_matrix.locations_to_locations_distances[
                current_filtered_partial_route[-1]][potential_next_location_id]

        if current_distance < min_distance:
            min_distance = current_distance
            next_courier_id = potential_next_courier_id

    return next_courier_id


def choose_next_location_id(
        problem_state: ProblemState,
        routes: Routes,
        next_courier_id: CourierId
) -> LocationId:
    courier_vehicle_state = [vehicle_state for vehicle_state in problem_state.vehicle_states
                             if vehicle_state.courier_id == next_courier_id][0]
    filtered_partial_route = courier_vehicle_state.get_filtered_partial_route()
    courier_route = [route for route in routes.routes if route.vehicle_id == next_courier_id][0].location_ids
    assert len(filtered_partial_route) < len(courier_route)
    for j in range(len(filtered_partial_route)):
        assert filtered_partial_route[j] == courier_route[j]
    next_location_id = courier_route[len(filtered_partial_route)]
    return next_location_id
