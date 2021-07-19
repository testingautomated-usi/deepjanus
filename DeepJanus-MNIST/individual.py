import json
from os import makedirs
from os.path import join, exists

from numpy import mean

import evaluator
from folder import Folder
from config import EXPLABEL


class Individual:
    # Global counter of all the individuals (it is increased each time an individual is created or mutated).
    COUNT = 0
    SEEDS = set()

    def __init__(self, member1, member2):
        self.id = Individual.COUNT
        self.seed = None
        self.members_distance = None
        self.sparseness = None
        self.misclass = None
        self.aggregate_ff = None
        self.archive_candidate = None
        self.m1 = member1
        self.m2 = member2

    def reset(self):
        self.id = Individual.COUNT
        self.members_distance = None
        self.sparseness = None
        self.misclass = None
        self.aggregate_ff = None
        self.archive_candidate = None

    def to_dict(self):
        return {'id': str(self.id),
                'seed': str(self.seed),
                #TODO: expected label depending on member
                'expected_label': str(EXPLABEL),
                'm1': str(self.m1.id),
                'm2': str(self.m2.id)
        }

    def export(self):
        if not exists(Folder.DST_IND):
            makedirs(Folder.DST_IND)
        dst = join(Folder.DST_IND, "ind"+str(self.id))
        data = self.to_dict()
        filedest = dst + ".json"
        with open(filedest, 'w') as f:
            (json.dump(data, f, sort_keys=True, indent=4))

    def evaluate(self, archive):
        self.sparseness = None

        if self.misclass is None:
            # Calculate fitness function 2
            self.misclass = evaluator.evaluate_ff2(self.m1.confidence,
                                                   self.m2.confidence)

            self.archive_candidate = (self.m1.correctly_classified !=
                                      self.m2.correctly_classified)

        if self.members_distance is None:
            # Calculate fitness function 1
            self.members_distance = evaluator.evaluate_ff1(self.m1.purified,
                                                           self.m2.purified)

        # Recalculate sparseness at each iteration
        self.sparseness = evaluator.evaluate_sparseness(self, archive)
        if self.sparseness == 0.0:
            print(self.sparseness)
            print("BUG")

        self.aggregate_ff = evaluator.evaluate_aggregate_ff(self.sparseness,
                                                            self.members_distance)

        return self.aggregate_ff, self.misclass

    def mutate(self):
        raise NotImplemented()

    def distance(self, i2):
        i1 = self
        a = i1.m1.distance(i2.m1)
        b = i1.m1.distance(i2.m2)
        c = i1.m2.distance(i2.m1)
        d = i1.m2.distance(i2.m2)

        dist = mean([min(a, b), min(c, d), min(a, c), min(b, d)])
        return dist
