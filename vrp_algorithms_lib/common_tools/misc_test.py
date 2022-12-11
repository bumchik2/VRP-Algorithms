import vrp_algorithms_lib.common_tools.misc as misc
import numpy as np


def test_get_geodesic_speed_km_h():
    assert misc.get_geodesic_speed_km_h() == 18


def test_get_euclidean_distance_km():
    # Distance in kilometers between Lyon and Paris
    assert np.isclose(misc.get_euclidean_distance_km(45.7597, 4.8422, 48.8567, 2.3508), 392.2172595594006)
