from abc import ABC
from abc import abstractmethod
from typing import List


class BaseClusterizer(ABC):
    @abstractmethod
    def clusterize(self, points: List[List[float]], number_of_clusters: int) -> List[int]:
        raise NotImplementedError('clusterize is not implemented for BaseClusterizer')
