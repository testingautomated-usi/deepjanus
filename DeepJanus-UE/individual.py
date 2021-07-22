import evaluator
from predictor import Predictor


class Individual:
    # Global counter of all the individuals (it is increased each time an individual is created or mutated).
    COUNT = 0
    SEEDS = set()

    def __init__(self, member1, member2, seed):
        self.id = Individual.COUNT
        self.seed = seed
        self.distance = None
        self.sparseness = None
        self.misclass = None
        self.aggregate_ff = None
        self.misbehaviour = None
        self.member1 = member1
        self.member2 = member2

    def reset(self):
        self.id =Individual.COUNT
        self.distance = None
        self.sparseness = None
        self.misclass = None
        self.aggregate_ff = None
        self.misbehaviour = None

    def evaluate(self, archive):
        self.sparseness = None

        if self.misclass is None:
            # Calculate fitness function 2
            self.misclass = evaluator.evaluate_ff2(self.member1.diff,
                                                   self.member2.diff)

            self.misbehaviour = self.member1.correctly_classified != self.member2.correctly_classified

        if self.distance is None:
            # Calculate fitness function 1
            self.distance = evaluator.evaluate_ff1(self.member1.model_params,
                                                   self.member2.model_params)

        # Recalculate sparseness at each iteration
        self.sparseness = evaluator.evaluate_sparseness(self, archive)
        if self.sparseness == 0.0:
            print(self.sparseness)
            print("BUG")

        self.aggregate_ff = evaluator.evaluate_aggregate_ff(self.sparseness, self.distance)

        return self.aggregate_ff, self.misclass
