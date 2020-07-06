from core.problem import Problem
from core.member import Member


class SeedPool:
    def __init__(self, problem: Problem):
        self.problem = problem

    def __len__(self):
        raise NotImplemented()

    def __getitem__(self, item) -> Member:
        raise NotImplemented()
