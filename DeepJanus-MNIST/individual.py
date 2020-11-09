import evaluator
from predictor import Predictor


class Individual:
    # Global counter of all the individuals (it is increased each time an individual is created or mutated).
    COUNT = 0
    SEEDS = set()

    def __init__(self, member1, member2):
        self.id = Individual.COUNT
        self.seed = None
        self.distance = None
        self.sparseness = None
        self.misclass = None
        self.aggregate_ff = None
        self.member1 = member1
        self.member2 = member2

    def reset(self):
        self.id = Individual.COUNT
        self.distance = None
        self.sparseness = None
        self.misclass = None
        self.aggregate_ff = None

    def evaluate(self, archive):
        self.sparseness = None

        if self.misclass is None:
            # predicted_label1, confidence1 = \
            #    Predictor.predict_single(self.member1.purified, self.member1.expected_label)
            #
            # predicted_label2, confidence2 = \
            #    Predictor.predict_single(self.member2.purified, self.member2.expected_label)
            #
            # import numpy as np
            # assert(np.abs(self.member1.confidence - confidence1) < 0.01)
            # assert (np.abs(self.member2.confidence - confidence2) < 0.01)


            #self.member1.predicted_label, self.member1.confidence = \
            #    Predictor.predict(self.member1.purified, self.member1.expected_label)

            #self.member2.predicted_label, self.member2.confidence = \
            #    Predictor.predict(self.member2.purified, self.member2.expected_label)

            # Calculate fitness function 2
            self.misclass = evaluator.evaluate_ff2(self.member1.confidence,
                                                   self.member2.confidence)

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
