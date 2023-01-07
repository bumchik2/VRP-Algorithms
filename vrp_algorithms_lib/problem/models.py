from typing import Dict
from typing import List
from typing import NewType
from typing import Set
from typing import Tuple
from typing import Union


from pydantic import BaseModel

from vrp_algorithms_lib.common_tools.misc import get_euclidean_distance_km, get_geodesic_speed_km_h

LocationId = NewType('LocationId', str)
DepotId = NewType('DepotId', str)
CourierId = NewType('CourierId', str)


class Point(BaseModel):
    lat: float
    lon: float


class Depot(BaseModel):
    id: DepotId
    point: Point


class Courier(BaseModel):
    id: CourierId


class Location(BaseModel):
    id: LocationId
    depot_id: DepotId
    point: Point
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

    def get_depot(
            self
    ) -> Depot:
        assert len(self.depots) == 1
        return self.depots[list(self.depots.keys())[0]]


class Route(BaseModel):
    vehicle_id: CourierId
    location_ids: List[LocationId]

    def get_edges_set(
            self,
            depot_id: DepotId
    ) -> Set[Tuple[Union[LocationId, DepotId], LocationId]]:
        assert depot_id not in self.location_ids

        edges_set = set()

        if len(self.location_ids) == 0:
            return edges_set

        edges_set.add((depot_id, self.location_ids[0]))

        for i in range(0, len(self.location_ids) - 1):
            location_id_1 = self.location_ids[i]
            location_id_2 = self.location_ids[i + 1]
            edges_set.add((location_id_1, location_id_2))

        return edges_set


class Routes(BaseModel):
    routes: List[Route]

    def get_route_by_vehicle_id(
            self,
            vehicle_id: CourierId
    ) -> Route:
        route_wrapped = [route for route in self.routes if route.vehicle_id == vehicle_id]
        assert len(route_wrapped) == 1
        return route_wrapped[0]

    def get_edges_set(
            self,
            depot_id: DepotId
    ) -> Set[Tuple[Union[LocationId, DepotId], LocationId]]:
        edges_set = set()
        for route in self.routes:
            edges_set.update(route.get_edges_set(depot_id))
        assert len(edges_set) == sum([len(route.location_ids) for route in self.routes])
        return edges_set


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
                lat_1=depot.point.lat, lon_1=depot.point.lon, lat_2=location.point.lat, lon_2=location.point.lon
            )

    for location_1 in locations.values():
        locations_to_locations_distances[location_1.id] = {}
        for location_2 in locations.values():
            locations_to_locations_distances[location_1.id][location_2.id] = get_euclidean_distance_km(
                lat_1=location_1.point.lat, lon_1=location_1.point.lon,
                lat_2=location_2.point.lat, lon_2=location_2.point.lon
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
