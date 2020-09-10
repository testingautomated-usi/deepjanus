import time
import argparse
from datetime import datetime
import numpy as np
import keras

# local imports
from mapelites import MapElites
from feature_dimension import FeatureDimension
import plot_utils
import utils
import vectorization_tools
from individual import Individual
from properties import BITMAP_THRESHOLD


class MapElitesMNIST(MapElites):

    def __init__(self, *args, **kwargs):
        super(MapElitesMNIST, self).__init__(*args, **kwargs)

    def map_x_to_b(self, x):
        """
        Map X solution to feature space dimension            
        :return: tuple of indexes
        """
        b = tuple()
        for ft in self.feature_dimensions:
            i = ft.feature_descriptor(self, x)               
            b = b + (i,)
        return b

    def performance_measure(self, x, archive):
        """
        Apply the fitness function to x
        """
        # "calculate performance measure"    
        pref = x.evaluate(archive)     
        return pref

    def generate_feature_dimensions(self, type): 
        fts = list()
        if type == 1:
            # feature 6: moves in svg path
            ft7 = FeatureDimension(name="Moves", feature_simulator="move_distance",bins=10)
            fts.append(ft7)

            # feature 2: Number of bitmaps above threshold
            ft2 = FeatureDimension(name="Bitmaps", feature_simulator="bitmap_count", bins=180)
            fts.append(ft2)

        elif type == 2:
            # feature 6: moves in svg path
            ft7 = FeatureDimension(name="Moves", feature_simulator="move_distance",bins=10)
            fts.append(ft7)

            # feature 7: orientation
            ft8 = FeatureDimension(name="Orientation", feature_simulator="orientation_calc",bins=100)
            fts.append(ft8)
        
        else:
            # feature 7: orientation
            ft8 = FeatureDimension(name="Orientation", feature_simulator="orientation_calc",bins=100)
            fts.append(ft8)

            # feature 2: Number of bitmaps above threshold
            ft2 = FeatureDimension(name="Bitmaps", feature_simulator="bitmap_count", bins=180)
            fts.append(ft2)
               

        return fts

    def feature_simulator(self, function, x):
        """
        Calculates the number of control points of x's svg path/number of bitmaps above threshold
        :param x: genotype of candidate solution x
        :return: 
        """
        if function == 'bitmap_count':
            return utils.bitmap_count(x.member1, BITMAP_THRESHOLD)
        if function == 'move_distance':
            return utils.move_distance(x.member1)
        if function == 'orientation_calc':
            return utils.new_orientation_calc(x.member1,0)
            
