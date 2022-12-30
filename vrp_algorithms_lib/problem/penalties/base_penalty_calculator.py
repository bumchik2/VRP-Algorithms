from vrp_algorithms_lib.problem.models import ProblemDescription, Routes


class BasePenaltyCalculator:
    @staticmethod
    def get_penalty_name(
    ) -> str:
        raise NotImplementedError

    def calculate(
            self,
            problem_description: ProblemDescription,
            routes: Routes
    ) -> float:
        raise NotImplementedError
