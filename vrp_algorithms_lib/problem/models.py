from typing import Dict
from typing import NewType
from typing import List

from pydantic import BaseModel

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
    depot_id: str
    point: Point
    time_window_start_s: float
    time_window_end_s: float


class DistanceMatrix(BaseModel):
    depots_to_locations_distances: Dict[DepotId, Dict[LocationId, float]]
    locations_to_locations_distances: Dict[LocationId, Dict[LocationId, float]]


class TimeMatrix(BaseModel):
    depots_to_locations_travel_times: Dict[DepotId, Dict[LocationId, float]]
    locations_to_locations_travel_times: Dict[LocationId, Dict[LocationId, float]]


class PenaltyMultipliers(BaseModel):
    distance_penalty_multiplier: float


class ProblemDescription(BaseModel):
    locations: Dict[LocationId, Location]
    couriers: Dict[CourierId, Courier]
    depots: Dict[DepotId, Depot]
    distance_matrix: DistanceMatrix
    time_matrix: TimeMatrix
    penalties: PenaltyMultipliers


class Route(BaseModel):
    vehicle_id: CourierId
    location_ids: List[LocationId]


class Routes(BaseModel):
    routes: List[Route]
