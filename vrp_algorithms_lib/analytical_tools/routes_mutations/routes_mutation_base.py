from abc import ABC, abstractmethod
from vrp_algorithms_lib.problem.models import Routes


class RoutesMutationBase(ABC):
    @abstractmethod
    def modify_routes(
            self,
            routes: Routes
    ) -> None:
        raise NotImplementedError
