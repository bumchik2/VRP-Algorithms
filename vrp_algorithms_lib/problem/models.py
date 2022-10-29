from pydantic import BaseModel
from typing import Dict
from typing import NewType

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
    depot_id: str
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


class PenaltyMultipliers(BaseModel):
    distance_penalty_multiplier: float


class ProblemDescription(BaseModel):
    locations: Dict[LocationId, Location]
    couriers: Dict[CourierId, Courier]
    depots: Dict[DepotId, Depot]
    distance_matrix: DistanceMatrix
    time_matrix: TimeMatrix
    penalties: PenaltyMultipliers
