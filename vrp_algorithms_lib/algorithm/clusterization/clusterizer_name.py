from enum import Enum
from typing import NewType

from vrp_algorithms_lib.algorithm.clusterization.kmeans_clusterizer import KMeansClusterizer


class ClusterizerName(Enum):
    T = NewType('ClusterizerName', str)

    KMeansClusterizer = T('KMeansClusterizer')


CLUSTERIZER_NAME_TO_CLUSTERIZER_TYPE = {
    ClusterizerName.KMeansClusterizer: KMeansClusterizer
}
