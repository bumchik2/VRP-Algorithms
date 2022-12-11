from haversine import haversine


def get_euclidean_distance_km(lat_1: float, lon_1: float, lat_2: float, lon_2: float):
    distance = haversine((lat_1, lon_1), (lat_2, lon_2))
    return distance


def get_geodesic_speed_km_h():
    return 18.
