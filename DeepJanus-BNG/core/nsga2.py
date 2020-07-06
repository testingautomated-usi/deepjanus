import random

import numpy
from deap import base
from deap import creator
from deap import tools

from core.log_setup import get_logger
from core.problem import Problem

log = get_logger(__file__)


def main(problem: Problem = None, seed=None):
    config = problem.config
    random.seed(seed)

    # DEAP framework setup
    # We define a bi-objective fitness function.
    # 1. Maximize the sparseness minus an amount due to the distance between members
    # 2. Minimize the distance to the decision boundary
    creator.create("FitnessMulti", base.Fitness, weights=config.fitness_weights)
    creator.create("Individual", problem.deap_individual_class(), fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    problem.toolbox = toolbox
    # We need to define the individual, the evaluation function (OOBs), mutation
    # toolbox.register("individual", tools.initIterate, creator.Individual)
    toolbox.register("individual", problem.deap_generate_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", problem.deap_evaluate_individual)
    toolbox.register("mutate", problem.deap_mutate_individual)
    toolbox.register("select", tools.selNSGA2)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "min", "max", "avg", "std"

    # Generate initial population.
    log.info("### Initializing population....")
    pop = toolbox.population(n=config.POPSIZE)

    # Evaluate the initial population.
    # Note: the fitness functions are all invalid before the first iteration since they have not been evaluated.
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]

    problem.pre_evaluate_members(invalid_ind)

    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    problem.archive.process_population(pop)

    # This is just to assign the crowding distance to the individuals (no actual selection is done).
    pop = toolbox.select(pop, len(pop))

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # Initialize the archive.
    problem.on_iteration(0, pop, logbook)

    # Begin the generational process
    for gen in range(1, config.NUM_GENERATIONS):
        # invalid_ind = [ind for ind in pop]


        # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        # for ind, fit in zip(invalid_ind, fitnesses):
        #    ind.fitness.values = fit

        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [ind.clone() for ind in offspring]

        problem.reseed(pop, offspring)

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        to_eval = offspring + pop
        invalid_ind = [ind for ind in to_eval]

        problem.pre_evaluate_members(invalid_ind)

        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        problem.archive.process_population(offspring + pop)
        # Select the next generation population
        pop = toolbox.select(pop + offspring, config.POPSIZE)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)
        problem.on_iteration(gen, pop, logbook)
    return pop, logbook


if __name__ == "__main__":
    final_population, search_stats = main()
