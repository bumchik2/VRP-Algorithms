from vrp_algorithms_lib.problem.models import ProblemDescription, Routes


class BasePenaltyCalculator:
    def calculate(
            self,
            problem_description: ProblemDescription,
            routes: Routes
    ) -> float:
        raise NotImplementedError
