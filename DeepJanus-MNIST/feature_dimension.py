
class FeatureDimension:
    """
    Implements a feature dimension of the MAP-Elites algorithm
    """

    def __init__(self, name, feature_simulator, bins):
        """
        :param name: Name of the feature dimension
        :param bins: Array of bins, from starting value to last value of last bin
        """
        self.name = name
        self.feature_simulator = feature_simulator
        self.bins = bins

    def feature_descriptor(self, mapelite, x):
        """
        Simulate the candidate solution x and record its feature descriptor
        :param x: genotype of candidate solution x
        :return:
        """
        i = mapelite.feature_simulator(self.feature_simulator,x)
        return i



