import itertools
import random
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from deap import base, creator, tools
from deap.tools.emo import selNSGA2
import keras
import json
import os
import sys

# local imports
import vectorization_tools
from digit_input import Digit
from digit_mutator import DigitMutator
import archive_manager
from individual import Individual
from properties import NGEN, IMG_SIZE, \
    EXPECTED_LABEL, INITIALPOP, \
    ORIGINAL_SEEDS, RESEEDUPPERBOUND, GENERATE_ONE_ONLY, RUNTIME, INTERVAL, POPSIZE
from mapelites_mnist import MapElitesMNIST
import utils
import plot_utils

# Load the dataset.
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Fetch the starting seeds from file
with open(ORIGINAL_SEEDS) as f:
    starting_seeds = f.read().split(',')[:-1]
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
    xml_desc = vectorization_tools.vectorize(seed_image)
    return Digit(xml_desc, EXPECTED_LABEL)


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


def run(dir_name, rand_seed=None):
    random.seed(rand_seed)
    start_time = datetime.now()
    starttime = time.time()

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

    ii = 1
    # Begin the generational process
    for gen in range(1, NGEN):
        elapsed_time = datetime.now() - start_time
        if elapsed_time.seconds <= RUNTIME:
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
            fitnesses = [toolbox.evaluate(i, archive.get_archive()) for i in invalid_ind]

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            for ind in population + offspring:
                if ind.fitness.values[1] < 0:
                    archive.update_archive(ind)

            # Select the next generation population
            population = toolbox.select(population + offspring, POPSIZE)

            # Generate maps
            elapsed_time = datetime.now() - start_time
            if (elapsed_time.seconds) >= INTERVAL*ii:
                print("generating map")                
                #archive.create_report(x_test, Individual.SEEDS, gen)
                generate_maps((INTERVAL*ii/60), gen, dir_name)
                ii += 1

            # Update the statistics with the new population
            record = stats.compile(population)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)
            print(logbook.stream)

    print(logbook.stream)

    endtime = time.time()
    elapsedtime = endtime - starttime
    print(f"Running time {time.strftime('%H:%M:%S', time.gmtime(elapsedtime))}")        

    return population


def generate_maps(execution_time, iterations, dir_name):    
    # The experiment folder
    now = datetime.now().strftime("%Y%m%d%H%M%S")    
    log_dir_name = "log_"+str(POPSIZE)+"_"+str(iterations)+"_"+str(execution_time)+"_"+str(now) 
    log_dir_path = Path('logs/'+str(dir_name)+"/"+str(log_dir_name))
    log_dir_path.mkdir(parents=True, exist_ok=True)   
    if len(archive.get_archive()) > 0:
        ''' type #1 : Moves & Bitmaps
            type #2 : Moves & Orientation
            type #3 : Orientation & Bitmaps
        '''
        for i in range(1,4):
            map_E = MapElitesMNIST(i, NGEN, POPSIZE, True, log_dir_path)               
            image_dir_path = Path(f'logs/{dir_name}/{log_dir_name}/{map_E.feature_dimensions[1].name}_{map_E.feature_dimensions[0].name}')
            image_dir_path.mkdir(parents=True, exist_ok=True)
            for ind in archive.get_archive():             
                map_E.place_in_mapelites(ind, archive.get_archive())

            # rescale        
            map_E.solutions, map_E.performances = utils.rescale(map_E.solutions,map_E.performances)     

            # filled values                                 
            filled = np.count_nonzero(map_E.solutions!=None)
            total = np.size(map_E.solutions)
            filled_density = (filled / total)
                
            Individual.COUNT_MISS = 0    
            covered_seeds = set()
            mis_seeds = set()
            for (i,j), value in np.ndenumerate(map_E.performances): 
                if map_E.performances[i,j] != np.inf:
                    covered_seeds.add(map_E.solutions[i,j].seed)
                    if map_E.performances[i,j] < 0: 
                        mis_seeds.add(map_E.solutions[i,j].seed)
                        Individual.COUNT_MISS += 1
                        utils.print_image(f"{image_dir_path}/({i},{j})", map_E.solutions[i,j].member1.purified)
                    else:
                        utils.print_image(f"{image_dir_path}/({i},{j})", map_E.solutions[i,j].member1.purified, 'gray')

            report = {               
                'Covered seeds' : len(covered_seeds),
                'Filled cells': str(filled),
                'Filled density': str(filled_density),
                'Misclassified seeds': len(mis_seeds),
                'Misclassification': str(Individual.COUNT_MISS),
                'Misclassification density': str(Individual.COUNT_MISS/filled)
                
            }  
            dst = f"logs/{dir_name}/report_"+ map_E.feature_dimensions[1].name +'_'+ map_E.feature_dimensions[0].name+ '_'+ str(execution_time) +'.json'
            report_string = json.dumps(report)

            file = open(dst, 'w')
            file.write(report_string)
            file.close()

            map_E.plot_map_of_elites()
            plot_utils.plot_fives(f"logs/{dir_name}/{log_dir_name}", map_E.feature_dimensions[1].name, map_E.feature_dimensions[0].name)  
          


if __name__ == "__main__":    
    archive = archive_manager.Archive()    
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    dir_name = f"temp_{now}" 
    pop = run(dir_name)
    
    filename = f"logs/{dir_name}/results_{now}"
    utils.generate_reports(filename, f"logs/{dir_name}")

