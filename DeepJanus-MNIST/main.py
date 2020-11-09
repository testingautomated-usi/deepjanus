import random

import vectorization_tools
from digit_input import Digit
from digit_mutator import DigitMutator
from predictor import Predictor
from utils import print_archive

import numpy as np
from deap import base, creator, tools
from deap.tools.emo import selNSGA2
from tensorflow import keras

import h5py

import archive_manager
from individual import Individual
from properties import NGEN, \
    POPSIZE, INITIALPOP, \
    RESEEDUPPERBOUND, GENERATE_ONE_ONLY, DATASET

# Load the dataset.
hf = h5py.File(DATASET, 'r')
x_test = hf.get('xn')
x_test = np.array(x_test)
y_test = hf.get('yn')
y_test = np.array(y_test)

# Fetch the starting seeds from file
starting_seeds = [i for i in range(len(y_test))]
random.shuffle(starting_seeds)
starting_seeds = starting_seeds[:POPSIZE]

# DEAP framework setup.
toolbox = base.Toolbox()
# Define a bi-objective fitness function.
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
# Define the individual.
creator.create("Individual", Individual, fitness=creator.FitnessMulti)


def generate_digit(seed):
    seed_image = x_test[int(seed)]
    label = y_test[int(seed)]
    xml_desc = vectorization_tools.vectorize(seed_image)
    return Digit(xml_desc, label)


def generate_individual():
    Individual.COUNT += 1

    if INITIALPOP == 'random':
        # Choose randomly a file in the original dataset.
        seed = random.choice(starting_seeds)
        Individual.SEEDS.add(seed)
    elif INITIALPOP == 'seeded':
        # Choose sequentially the inputs from the seed list.
        # NOTE: number of seeds should be no less than the initial population
        assert (len(starting_seeds) == POPSIZE)
        seed = starting_seeds[Individual.COUNT - 1]
        Individual.SEEDS.add(seed)

    if not GENERATE_ONE_ONLY:
        digit1, digit2, distance_inputs = DigitMutator(generate_digit(seed)).generate()
    else:
        digit1 = generate_digit(seed)
        digit2 = digit1.clone()
        distance_inputs = DigitMutator(digit2).mutate()

    # TODO: do not have info about the label
    individual = creator.Individual(digit1, digit2)
    individual.distance = distance_inputs
    individual.seed = seed

    return individual


def reseed_individual(seeds):
    Individual.COUNT += 1
    # Chooses randomly the seed among the ones that are not covered by the archive
    if len(starting_seeds) > len(seeds):
        chosen_seed = random.sample(set(starting_seeds) - seeds, 1)[0]
    else:
        chosen_seed = random.choice(starting_seeds)

    if not GENERATE_ONE_ONLY:
        first_digit, second_digit, distance_inputs = DigitMutator(generate_digit(chosen_seed)).generate()
    else:
        digit1 = generate_digit(chosen_seed)
        digit2 = digit1.clone()
        distance_inputs = DigitMutator(digit2).mutate()

    individual = creator.Individual(first_digit, second_digit)
    individual.distance = distance_inputs
    individual.seed = chosen_seed
    return individual


# Evaluate an individual.
def evaluate_individual(individual, current_solution):
    individual.evaluate(current_solution)
    return individual.aggregate_ff, individual.misclass


def mutate_individual(individual):
    Individual.COUNT += 1
    # Select one of the two members of the individual.
    if random.getrandbits(1):
        distance_inputs = DigitMutator(individual.member1).mutate(reference=individual.member2)
    else:
        distance_inputs = DigitMutator(individual.member2).mutate(reference=individual.member1)
    individual.reset()
    individual.distance = distance_inputs


toolbox.register("individual", generate_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_individual)
toolbox.register("select", selNSGA2)
toolbox.register("mutate", mutate_individual)


def pre_evaluate_batch(invalid_ind):
    batch_members = [i.member1 for i in invalid_ind if i.member1.predicted_label is None]
    batch_members += [i.member2 for i in invalid_ind if i.member2.predicted_label is None]

    batch_img = [m.purified for m in batch_members]
    batch_img = np.reshape(batch_img, (-1, 28, 28, 1))

    batch_label = ([m.expected_label for m in batch_members])

    #batch_seed = [m.seed for m in batch_members]

    predictions, confidences = (Predictor.predict(img=batch_img, label=batch_label))

    for member, confidence, prediction in zip(batch_members, confidences, predictions):
        #member.correctly_classified = correct
        member.confidence = confidence
        member.predicted_label = prediction
        if member.expected_label == member.predicted_label:
            member.correctly_classified = True
        else:
            member.correctly_classified = False


def main(rand_seed=None):
    random.seed(rand_seed)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "min", "max", "avg", "std"

    # Generate initial population.
    print("### Initializing population ....")
    population = toolbox.population(n=POPSIZE)

    # Evaluate the individuals with an invalid fitness.
    # Note: the fitnesses are all invalid before the first iteration since they have not been evaluated
    invalid_ind = [ind for ind in population]
    to_evaluate_ind = [ind for ind in population if ind.misclass is None]

    pre_evaluate_batch(to_evaluate_ind)


    # Note: the sparseness is calculated wrt the archive. It can be calculated wrt population+archive
    # Therefore, we pass to the evaluation method the current archive.
    fitnesses = [toolbox.evaluate(i, archive.get_archive()) for i in invalid_ind]
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Update archive with the individuals on the decision boundary.
    for ind in population:
        if ind.misclass < 0:
            archive.update_archive(ind)

    print("### Number of Individuals generated in the initial population: " + str(Individual.COUNT))

    # This is just to assign the crowding distance to the individuals (no actual selection is done).
    population = toolbox.select(population, len(population))

    record = stats.compile(population)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN):
        # Vary the population.
        offspring = tools.selTournamentDCD(population, len(population))
        offspring = [toolbox.clone(ind) for ind in offspring]

        # Reseeding
        if len(archive.get_archive()) > 0:
            seed_range = random.randrange(1, RESEEDUPPERBOUND)
            candidate_seeds = archive.archived_seeds
            for i in range(len(population)):
                if population[i].seed in archive.archived_seeds:
                    population[i] = reseed_individual(candidate_seeds)

        # Mutation.
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals
        # NOTE: all individuals in both population and offspring are evaluated to assign crowding distance.
        invalid_ind = [ind for ind in population + offspring]
        pre_evaluate_batch(invalid_ind)

        fitnesses = [toolbox.evaluate(i, archive.get_archive()) for i in invalid_ind]

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        for ind in population + offspring:
            if ind.fitness.values[1] < 0:
                archive.update_archive(ind)

        # Select the next generation population
        population = toolbox.select(population + offspring, POPSIZE)

        if gen % 300 == 0:
            archive.create_report(x_test, Individual.SEEDS, gen)

        # Update the statistics with the new population
        record = stats.compile(population)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

    print(logbook.stream)

    return population


if __name__ == "__main__":
    archive = archive_manager.Archive()
    pop = main()

    print_archive(archive.get_archive())
    archive.create_report(x_test, Individual.SEEDS, 'final')
    print("GAME OVER")
