from vrp_algorithms_lib.problem.models import Point
from abc import ABC
from abc import abstractmethod
from typing import List


class BaseClusterizer(ABC):
    @abstractmethod
    def clusterize(self, points: List[Point], number_of_clusters: int) -> List[int]:
        raise NotImplementedError('clusterize is not implemented for BaseClusterizer')
