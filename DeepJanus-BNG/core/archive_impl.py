from itertools import permutations
from typing import List, Tuple
import logging as log

from core.archive import Archive
from core.individual import Individual
from core.misc import closest_elements


class GreedyArchive(Archive):
    def process_population(self, pop: List[Individual]):
        for candidate in pop:
            if candidate.oob_ff < 0:
                self.add(candidate)


class SmartArchive(Archive):
    def __init__(self, ARCHIVE_THRESHOLD):
        super().__init__()
        self.ARCHIVE_THRESHOLD = ARCHIVE_THRESHOLD

    def process_population(self, pop: List[Individual]):
        for candidate in pop:
            assert candidate.oob_ff, candidate.name
            if candidate.oob_ff < 0:
                if len(self) == 0:
                    self._int_add(candidate)
                    log.debug('add initial individual')
                else:
                    # uses semantic_distance to exploit behavioral information
                    closest_archived, candidate_archived_distance = \
                        closest_elements(self, candidate, lambda a, b: a.semantic_distance(b))[0]
                    closest_archived: Individual

                    if candidate_archived_distance > self.ARCHIVE_THRESHOLD:
                        log.debug('candidate is far from any archived individual')
                        self._int_add(candidate)
                    else:
                        log.debug('candidate is very close to an archived individual')
                        if candidate.members_distance < closest_archived.members_distance:
                            log.debug('candidate is better than archived')
                            self._int_add(candidate)
                            self.remove(closest_archived)
                            print('archive rem', closest_archived)
                        else:
                            log.debug('archived is better than candidate')

    def _int_add(self, candidate):
        self.add(candidate)
        print('archive add', candidate)


class PopArchive(Archive):
    def __init__(self, ARCHIVE_THRESHOLD):
        super().__init__()
        self.ARCHIVE_THRESHOLD = ARCHIVE_THRESHOLD
        self.pop = None

    def update_pop(self, pop: List[Individual]):
        self.pop = pop

    def process_population(self, pop: List[Individual]):
        for candidate in pop:
            assert candidate.oob_ff, candidate.name
            if candidate.oob_ff < 0:
                if len(self) == 0:
                    self._int_add(candidate)
                    log.debug('add initial individual')
                else:
                    # uses semantic_distance to exploit behavioral information
                    closest_archived, candidate_archived_distance = \
                        closest_elements(self, candidate, lambda a, b: a.semantic_distance(b))[0]
                    closest_archived: Individual

                    if candidate_archived_distance > self.ARCHIVE_THRESHOLD:
                        log.debug('candidate is far from any archived individual')
                        self._int_add(candidate)
                    else:
                        log.debug('candidate is very close to an archived individual')
                        if candidate.members_distance < closest_archived.members_distance:
                            log.debug('candidate is better than archived')
                            self._int_add(candidate)
                            self.remove(closest_archived)
                            print('archive rem', closest_archived)
                        else:
                            log.debug('archived is better than candidate')

    def _int_add(self, candidate):
        self.add(candidate)
        print('archive add', candidate)



class SlowArchive(Archive):
    def __init__(self, ARCHIVE_THRESHOLD):
        super().__init__()
        self.ARCHIVE_THRESHOLD = ARCHIVE_THRESHOLD

    def process_population(self, pop: List[Individual]):
        for candidate in pop:
            assert candidate.oob_ff, candidate.name
            if candidate.oob_ff < 0:
                if len(self) == 0:
                    self._int_add(candidate)
                    log.debug('add initial individual')
                else:
                    # uses non-semantic_distance to compare individuals from the pop that are not on the boundary
                    closest_ind, candidate_distance = \
                        closest_elements(set(list(self)+pop), candidate, lambda a, b: a.distance(b))[0]
                    closest_ind: Individual

                    if candidate_distance > self.ARCHIVE_THRESHOLD:
                        log.debug('candidate is far from any archived individual')
                        self._int_add(candidate)
                    else:
                        log.debug('candidate is very close to an archived individual')
                        if candidate.members_distance < closest_ind.members_distance:
                            log.debug('candidate is better than archived')
                            self._int_add(candidate)
                            if closest_ind in self:
                                self.remove(closest_ind)
                            print('archive rem', closest_ind)
                        else:
                            log.debug('archived is better than candidate')

    def _int_add(self, candidate):
        self.add(candidate)
        print('archive add', candidate)


class RewardSparsenessArchive(Archive):
    def __init__(self):
        super().__init__()

    def process_population(self, pop: List[Individual]):
        candidates = [ind for ind in pop if ind.oob_ff < 0] + self
        ind: Individual
        perm: List[Tuple[Individual, Individual, float]] = [x + (x[0].distance(x[1]),) for x in
                                                            list(permutations(candidates, 2))]

        perm = sorted(perm, key=lambda a: a[2], reverse=True)
