from typing import List
from sklearn.cluster import KMeans
import numpy as np

from vrp_algorithms_lib.algorithm.clusterization.base_clusterizer import BaseClusterizer
from vrp_algorithms_lib.problem.models import Point


class KMeansClusterizer(BaseClusterizer):
    def clusterize(self, points: List[Point], number_of_clusters: int) -> List[int]:
        model = KMeans(n_clusters=number_of_clusters)

        x = np.array([
            [point.lon, point.lat] for point in points
        ])

        clusters = model.fit_predict(x)
        return clusters
