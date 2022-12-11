from typing import Dict
from typing import NewType
from typing import List

from pydantic import BaseModel

from vrp_algorithms_lib.common_tools.misc import get_euclidean_distance_km, get_geodesic_speed_km_h

LocationId = NewType('LocationId', str)
DepotId = NewType('DepotId', str)
CourierId = NewType('CourierId', str)


class Depot(BaseModel):
    id: DepotId
    lat: float
    lon: float


class Courier(BaseModel):
    id: CourierId


class Location(BaseModel):
    id: LocationId
    depot_id: DepotId
    lat: float
    lon: float
    time_window_start_s: float
    time_window_end_s: float


class DistanceMatrix(BaseModel):
    depots_to_locations_distances: Dict[DepotId, Dict[LocationId, float]]
    locations_to_locations_distances: Dict[LocationId, Dict[LocationId, float]]


class TimeMatrix(BaseModel):
    depots_to_locations_travel_times: Dict[DepotId, Dict[LocationId, float]]
    locations_to_locations_travel_times: Dict[LocationId, Dict[LocationId, float]]


class Penalties(BaseModel):
    distance_penalty_multiplier: float
    global_proximity_factor: float
    out_of_time_penalty_per_minute: float


class ProblemDescription(BaseModel):
    locations: Dict[LocationId, Location]
    couriers: Dict[CourierId, Courier]
    depots: Dict[DepotId, Depot]
    distance_matrix: DistanceMatrix
    time_matrix: TimeMatrix
    penalties: Penalties


class Route(BaseModel):
    vehicle_id: CourierId
    location_ids: List[LocationId]


class Routes(BaseModel):
    routes: List[Route]


def get_euclidean_distance_matrix(
        depots: Dict[DepotId, Depot],
        locations: Dict[LocationId, Location]
) -> DistanceMatrix:
    depots_to_locations_distances: Dict[DepotId, Dict[LocationId, float]] = {}
    locations_to_locations_distances: Dict[LocationId, Dict[LocationId, float]] = {}

    for depot in depots.values():
        depots_to_locations_distances[depot.id] = {}
        for location in locations.values():
            depots_to_locations_distances[depot.id][location.id] = get_euclidean_distance_km(
                lat_1=depot.lat, lon_1=depot.lon, lat_2=location.lat, lon_2=location.lon
            )

    for location_1 in locations.values():
        locations_to_locations_distances[location_1.id] = {}
        for location_2 in locations.values():
                locations_to_locations_distances[location_1.id][location_2.id] = get_euclidean_distance_km(
                    lat_1=location_1.lat, lon_1=location_1.lon, lat_2=location_2.lat, lon_2=location_2.lon
                )

    return DistanceMatrix(
        depots_to_locations_distances=depots_to_locations_distances,
        locations_to_locations_distances=locations_to_locations_distances
    )


def get_geodesic_time_matrix(
        depots: Dict[DepotId, Depot],
        locations: Dict[LocationId, Location],
        distance_matrix: DistanceMatrix
) -> TimeMatrix:
    depots_to_locations_travel_times: Dict[DepotId, Dict[LocationId, float]] = {}
    locations_to_locations_travel_times: Dict[LocationId, Dict[LocationId, float]] = {}
    geodesic_speed_km_h = get_geodesic_speed_km_h()

    for depot in depots.values():
        depots_to_locations_travel_times[depot.id] = {}
        for location in locations.values():
            depots_to_locations_travel_times[depot.id][location.id] = \
                distance_matrix.depots_to_locations_distances[depot.id][location.id] / geodesic_speed_km_h

    for location_1 in locations.values():
        locations_to_locations_travel_times[location_1.id] = {}
        for location_2 in locations.values():
            locations_to_locations_travel_times[location_1.id][location_2.id] = \
                distance_matrix.locations_to_locations_distances[location_1.id][location_2.id] / geodesic_speed_km_h

    return TimeMatrix(
        depots_to_locations_travel_times=depots_to_locations_travel_times,
        locations_to_locations_travel_times=locations_to_locations_travel_times
    )
