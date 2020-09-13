import time
import operator

import numpy as np
from datetime import datetime
from itertools import permutations
from abc import ABC, abstractmethod
from pathlib import Path

# local imports
from individual import Individual
from feature_dimension import FeatureDimension
from plot_utils import plot_heatmap
import utils


class MapElites(ABC):

    def __init__(self, type, iterations, bootstrap_individuals, minimization, dir_path):
        """
        :param iterations: Number of evolutionary iterations        
        :param bootstrap_individuals: Number of individuals randomly generated to bootstrap the algorithm       
        :param minimization: True if solving a minimization problem. False if solving a maximization problem.
        """
        self.elapsed_time = 0
        self.log_dir_path = dir_path
        self.minimization = minimization    
        # set the choice operator either to do a minimization or a maximization
        if self.minimization:
            self.place_operator = operator.lt
        else:
            self.place_operator = operator.ge

        self.iterations = iterations
        
        self.random_solutions = bootstrap_individuals
        
        self.feature_dimensions = self.generate_feature_dimensions(type)
        
        # get number of bins for each feature dimension
        ft_bins = [ft.bins for ft in self.feature_dimensions]     

        # Map of Elites: Initialize data structures to store solutions and fitness values
        self.solutions =  np.full(
            ft_bins, None,
            dtype=(object)
        )
        self.performances = np.full(ft_bins, np.inf, dtype=float)

        print("Configuration completed.")
   
    def place_in_mapelites(self, x, archive):
        """
        Puts a solution inside the N-dimensional map of elites space.
        The following criteria is used:

        - Compute the feature descriptor of the solution to find the correct
                cell in the N-dimensional space
        - Compute the performance of the solution
        - Check if the cell is empty or if the previous performance is worse
            - Place new solution in the cell
        :param x: genotype of an individual
        """
        # get coordinates in the feature space
        b = self.map_x_to_b(x)
        # performance of the x
        perf = self.performance_measure(x, archive)[1]
       
        reconstruct = False
        for i in range(len(b)):
            if b[i] >= self.feature_dimensions[i].bins:
                reconstruct = True
                self.feature_dimensions[i].bins = b[i] + 1      
        
        if reconstruct:
            self.recounstruct_map()

        # place operator performs either minimization or maximization
        if self.place_operator(perf, self.performances[b]):
            print(f"PLACE: Placing individual {x} at {b} with perf: {perf}")
            self.performances[b] = perf
            self.solutions[b] = x
        else:
            print(f"PLACE: Individual {x} rejected at {b} with perf: {perf} in favor of {self.performances[b]}")

    def recounstruct_map(self):
        """
        Extend Map of elites dynamically if needed
        """
        # get number of bins for each feature dimension
        ft_bins = [ft.bins for ft in self.feature_dimensions]

        new_solutions =  np.full(
            ft_bins, None,
            dtype=(object)
        )
        new_performances = np.full(ft_bins, np.inf, dtype=float)

        new_solutions[0:self.solutions.shape[0], 0:self.solutions.shape[1]] = self.solutions
        new_performances[0:self.performances.shape[0], 0:self.performances.shape[1]] = self.performances
        self.solutions = new_solutions
        self.performances = new_performances
        return

    def plot_map_of_elites(self):
        """
        Plot a heatmap of elites
        """
        plot_heatmap(self.performances,
                      self.feature_dimensions[1].name,
                      self.feature_dimensions[0].name,
                    savefig_path=self.log_dir_path,                     
                     )

    @abstractmethod
    def performance_measure(self, x):
        """
        Function to evaluate solution x and give a performance measure
        :param x: genotype of a solution
        :return: performance measure of that solution
        """
        pass

    @abstractmethod
    def map_x_to_b(self, x):
        """
        Function to map a solution x to feature space dimensions
        :param x: genotype of a solution
        :return: phenotype of the solution (tuple of indices of the N-dimensional space)
        """
        pass

    @abstractmethod
    def generate_feature_dimensions(self):
        """
        Generate a list of FeatureDimension objects to define the feature dimension functions
        :return: List of FeatureDimension objects
        """
        pass
