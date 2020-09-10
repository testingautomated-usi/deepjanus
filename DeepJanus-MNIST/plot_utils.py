
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from svgpathtools import svg2paths, wsvg
import os
import re
import numpy as np
import glob

def plot_heatmap(data, 
                xlabel,
                ylabel,                                   
                minimization=False,
                savefig_path=None,
                 ):
    
    ax = sns.heatmap(data)
    ax.invert_yaxis()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
  
    # get figure to save to file
    if savefig_path:
        ht_figure = ax.get_figure()
        ht_figure.savefig(f"{savefig_path}/heatmap_{xlabel}_{ylabel}.png", dpi=400)
    
    plt.clf()
    plt.cla()
    plt.close()
    

def plot_svg(xml, filename):
    root = ET.fromstring(xml)
    svg_path = root.find(NAMESPACE + 'path').get('d')
    wsvg(svg_path, filename=filename+'.svg')

def getImage(path):
    return OffsetImage(plt.imread(path))

def plot_fives(dir_path, xlabel, ylabel):    
    paths = glob.glob(dir_path + "/"+xlabel+"_"+ylabel+"/*.png")
    x=[]
    y=[]
    for a in paths:
        pattern = re.compile('([\d\.]+),([\d\.]+)')
        segments = pattern.findall(a)
        for se in segments: 
            x.append(int(se[0]))
            y.append(int(se[1]))

    plt.cla()
    
    fig, ax = plt.subplots(figsize=(10,10))
    #ax.scatter(x, y) 

    for x0, y0, path in zip(x, y,paths):
        ab = AnnotationBbox(getImage(path), (y0, x0), frameon=False)
        ax.add_artist(ab)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xticks(np.arange(-1, 25, 1)) 
    plt.yticks(np.arange(-1, 25,1)) 
    

    plt.grid(color='blue', linestyle='-', linewidth=0.09)
    ht_figure = ax.get_figure()
    ht_figure.savefig(f"{dir_path}/fives_{xlabel}_{ylabel}.png", dpi=400)
    
    plt.clf()    
    plt.close()