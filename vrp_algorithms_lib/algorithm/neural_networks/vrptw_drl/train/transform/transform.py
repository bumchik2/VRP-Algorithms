from abc import ABC, abstractmethod
from typing import Tuple

from vrp_algorithms_lib.problem.models import ProblemDescription, Routes


class Transform(ABC):
    @abstractmethod
    def __call__(
            self,
            problem_description: ProblemDescription,
            routes: Routes
    ) -> Tuple[ProblemDescription, Routes]:
        raise NotImplementedError
