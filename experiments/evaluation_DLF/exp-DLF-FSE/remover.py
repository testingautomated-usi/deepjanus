import glob
import os

import numpy as np
#from PIL import Image
import matplotlib.pyplot as plt
import imageio

from utils import get_diameter
from utils_csv import writeCsvLine

models = ["LQ", "HQ"]
runs = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]


for model in models:
    for run in runs:
        path = os.path.join(model, run)
        covered = set()

        filelist = [p for p in glob.glob(path + '/*.png') ]

        for f in filelist:
            toremove = [p for p in glob.glob(path + '/*.png') if "_in" in p]
            for filepath in toremove:
                try:
                    os.remove(filepath)
                except:
                    print("Error while deleting file : ", filepath)



