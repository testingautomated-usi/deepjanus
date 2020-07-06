import numpy as np
from tensorflow import keras

import evaluator
from predictor import Predictor
from properties import EXPECTED_LABEL, num_classes


class Individual:
    # Global counter of all the individuals (it is increased each time an individual is created or mutated).
    COUNT = 0
    SEEDS = set()

    def __init__(self, member1, member2):
        self.seed = None
        self.distance = None
        self.sparseness = None
        self.misclass = None
        self.aggregate_ff = None
        self.member1 = member1
        self.member2 = member2

    def reset(self):
        self.distance = None
        self.sparseness = None
        self.misclass = None
        self.aggregate_ff = None

    def evaluate(self, archive):
        self.sparseness = None

        if self.misclass is None:
            self.member1.predicted_label, self.member1.P_class, self.member1.P_notclass = \
                Predictor.predict(self.member1.purified)

            self.member2.predicted_label, self.member2.P_class, self.member2.P_notclass = \
                Predictor.predict(self.member2.purified)

            # Calculate fitness function 2
            self.misclass = evaluator.evaluate_ff2(self.member1.P_class,
                                                   self.member1.P_notclass,
                                                   self.member2.P_class,
                                                   self.member2.P_notclass)

        if self.distance is None:
            # Calculate fitness function 1
            self.distance = evaluator.evaluate_ff1(self.member1.purified,
                                                   self.member2.purified)

        # Recalculate sparseness at each iteration
        self.sparseness = evaluator.evaluate_sparseness(self, archive)
        if self.sparseness == 0.0:
            print(self.sparseness)
            print("BUG")

        self.aggregate_ff = evaluator.evaluate_aggregate_ff(self.sparseness, self.distance)

        return self.aggregate_ff, self.misclass
