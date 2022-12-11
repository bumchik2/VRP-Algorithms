from typing import List
from sklearn.cluster import KMeans
import numpy as np

from vrp_algorithms_lib.algorithm.clusterization.base_clusterizer import BaseClusterizer


class KMeansClusterizer(BaseClusterizer):
    def clusterize(self, points: List[List[float]], number_of_clusters: int) -> List[int]:
        model = KMeans(n_clusters=number_of_clusters)
        x = np.array(points)
        clusters = model.fit_predict(x)
        return clusters
