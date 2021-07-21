import random
import mutation_manager
import rasterization_tools
import vectorization_tools
from mnist_member import MnistMember
from config import MUTOPPROB
from utils import get_distance


class DigitMutator:

    def __init__(self, digit):
        self.digit = digit
        self.seed = digit.seed

    def mutate(self, reference=None):
        # Select mutation operator.
        rand_mutation_probability = random.uniform(0, 1)
        if rand_mutation_probability >= MUTOPPROB:
            mutation = 1
        else:
            mutation = 2

        condition = True
        counter_mutations = 0
        distance_inputs = 0
        while condition:
            counter_mutations += 1
            mutant_vector = mutation_manager.mutate(self.digit.xml_desc, mutation, counter_mutations/20)
            mutant_xml_desc = vectorization_tools.create_svg_xml(mutant_vector)
            rasterized_digit = rasterization_tools.rasterize_in_memory(mutant_xml_desc)

            distance_inputs = get_distance(self.digit.purified, rasterized_digit)

            if distance_inputs != 0:
                if reference is not None:
                    distance_inputs = get_distance(reference.purified, rasterized_digit)
                    if distance_inputs != 0:
                        condition = False
                else:
                    condition = False

        self.digit.xml_desc = mutant_xml_desc
        self.digit.purified = rasterized_digit
        self.digit.predicted_label = None
        self.digit.confidence = None
        self.digit.correctly_classified = None

        return distance_inputs

    def generate(self):
        # Select mutation operator.
        rand_mutation_probability = random.uniform(0, 1)
        if rand_mutation_probability >= MUTOPPROB:
            mutation = 1
        else:
            mutation = 2

        condition = True
        counter_mutations = 0
        distance_inputs = 0
        while condition:
            counter_mutations += 1
            vector1, vector2 = mutation_manager.generate(
                self.digit.xml_desc,
                mutation)
            v1_xml_desc = vectorization_tools.create_svg_xml(vector1)
            rasterized_digit1 = rasterization_tools.rasterize_in_memory(v1_xml_desc)

            v2_xml_desc = vectorization_tools.create_svg_xml(vector2)
            rasterized_digit2 = rasterization_tools.rasterize_in_memory(v2_xml_desc)

            distance_inputs = get_distance(rasterized_digit1,
                                           rasterized_digit2)

            if distance_inputs != 0:
                condition = False

        first_digit = MnistMember(v1_xml_desc,
                                  self.digit.expected_label,
                                  self.seed)
        second_digit = MnistMember(v2_xml_desc,
                                   self.digit.expected_label,
                                   self.seed)
        first_digit.purified = rasterized_digit1
        second_digit.purified = rasterized_digit2
        return first_digit, second_digit, distance_inputs


