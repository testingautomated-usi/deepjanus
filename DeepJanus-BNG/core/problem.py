from typing import List

from core.config import Config
from core.archive import Archive
from core.individual import Individual
from core.member import Member


class Problem:
    def __init__(self, config: Config, archive: Archive):
        self.config: Config = config
        self.archive = archive

    def deap_generate_individual(self) -> Individual:
        raise NotImplemented()

    def deap_mutate_individual(self, individual: Individual):
        individual.mutate()

    def deap_evaluate_individual(self, individual: Individual):
        raise NotImplemented()

    def deap_individual_class(self):
        raise NotImplemented()

    def on_iteration(self, idx, pop: List[Individual], logbook):
        raise NotImplemented()

    def member_class(self):
        raise NotImplemented()

    def reseed(self, population, offspring):
        raise NotImplemented()

    def generate_random_member(self) -> Member:
        raise NotImplemented()

    def pre_evaluate_members(self, individuals: List[Individual]):
        pass