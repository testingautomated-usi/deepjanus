from typing import List

from core.individual import Individual


class IndividualSet(set):
    pass


class Archive(IndividualSet):
    def process_population(self, pop: List[Individual]):
        raise NotImplemented()