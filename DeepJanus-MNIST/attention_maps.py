from ast import Break
import random
from matplotlib import gridspec
import numpy as np
import re

import vectorization_tools

from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np

from config import MODEL, MODEL2, num_classes

from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.gradcam import Gradcam
from tensorflow.keras.applications.imagenet_utils import preprocess_input

import matplotlib.patches as patches

import time
import rasterization_tools
from random import uniform
from config import MUTLOWERBOUND, MUTUPPERBOUND, MUTOFPROB

import gc
from sys import getsizeof

from predictor import Predictor

import potrace

from operator import itemgetter
import copy

from random import randint, uniform
import matplotlib.gridspec as gridspec

from timer import Timer
from os.path import exists, join
from os import makedirs

from PIL import Image
import glob

import csv
import os

score = CategoricalScore(0)
replace2linear = ReplaceToLinear()

# Attention_Technique = "Faster-ScoreCAM" #"Gradcam++"
Attention_Technique = "Gradcam++"
# Attention_Technique = "Gradcam"

if Attention_Technique == "Faster-ScoreCAM":

    # Create ScoreCAM object
    scorecam = Scorecam(Predictor.model, model_modifier=replace2linear)

elif Attention_Technique == "Gradcam++":

    # Create GradCAM++ object
    gradcam = GradcamPlusPlus(Predictor.model,
                        model_modifier=replace2linear,
                        clone=True)

# elif Attention_Technique == "Gradcam":

#     # Create Gradcam object
#     gradcam = Gradcam(Predictor.model,
#                     model_modifier=replace2linear,
#                     clone=True)

def input_reshape_images(x):
    # shape numpy vectors
    if keras.backend.image_data_format() == 'channels_first':
        x_reshape = x.reshape(x.shape[0], 1, 28, 28)
    else:
        x_reshape = x.reshape(x.shape[0], 28, 28, 1)
    x_reshape = x_reshape.astype('float32')
    x_reshape /= 255.0
    return x_reshape

def input_reshape_image(x):
    # shape numpy vectors
    if keras.backend.image_data_format() == 'channels_first':
        x_reshape = x.reshape(1, 1, 28, 28)
    else:
        x_reshape = x.reshape(1, 28, 28, 1)
    x_reshape = x_reshape.astype('float32')
    x_reshape /= 255.0
    return x_reshape

def get_attetion_region_and_darken_pixel_mth1(xai_image, orig_image, x_sqr_size, y_sqr_size):
    x_dim = xai_image.shape[0]
    y_dim = xai_image.shape[1]

    # print("x_dim ",x_dim)
    # print("y_dim ",y_dim)

    greater_value_sum_xai = 0
    x_final_pos = 0
    y_final_pos = 0

    for y_sqr_pos in range(y_dim - y_sqr_size):
        for x_sqr_pos in range(x_dim - x_sqr_size):
            sum_xai = 0
            for y_in_sqr in range(y_sqr_size):
                y_pixel_pos = y_sqr_pos + y_in_sqr
                for x_in_sqr in range(x_sqr_size):                    
                    x_pixel_pos = x_sqr_pos + x_in_sqr
                    if orig_image[y_pixel_pos][x_pixel_pos] > 0:
                        sum_xai += xai_image[y_pixel_pos][x_pixel_pos]
            if sum_xai > greater_value_sum_xai:
                greater_value_sum_xai = sum_xai
                x_final_pos = x_sqr_pos
                y_final_pos = y_sqr_pos

    return x_final_pos, y_final_pos


def AM_darken_attention_pixels_mth1(images, x_patch_size, y_patch_size, svg_path):
    """
    AM_get_attetion_svg_points_images_mth1 Iterate all the image looking for the region with more attention and return list of points (tuples) inside the square region with more attention.

    :param images: images should have the shape: (x, 28, 28) where x>=1
    :param x_patch_size: X size of the square region
    :param y_patch_size: Y size of the square region
    :param svg_path: A string with the digit's SVG path description. Ex: "M .... C .... Z".
    :return: A list of point positions that are inside the region found. A well detailed explanation about the structure of the list returned is described at the end of this function.
    """ 
    # start_time1 = time.time()
    xai = get_XAI_image(images)
    # start_time = time.time()
    # x, y = get_attetion_region(cam, images)
    list_of_ControlPointsInsideRegion = []
    for i in range(images.shape[0]):
        pattern = re.compile('([\d\.]+),([\d\.]+)\s[MCLZ]')
        ControlPoints = pattern.findall(svg_path)
        controlPoints = [(float(i[0]), float(i[1])) for i in ControlPoints]

        # x, y = get_attetion_region(xai[i], images[i], x_patch_size, y_patch_size) #Getting coordinates of the highest attetion region (patch) reference point
        x, y = get_attetion_region_and_darken_pixel_mth1(xai[i], images[i], x_patch_size, y_patch_size) #Getting coordinates of the highest attetion region (patch) reference point
        copy_orig_image = copy.deepcopy(images[i])
        for x_ite in range(x_patch_size):
            for y_ite in range(y_patch_size):
                copy_orig_image[y + y_ite][x + x_ite]=0
        # ControlPointsInsideRegion = getControlPointsInsideAttRegion(x,y,x_patch_size,y_patch_size, controlPoints) #Getting all the points inside the highest attetion patch
        # list_of_ControlPointsInsideRegion.append(ControlPointsInsideRegion)

    # end_time = time.time()

    # xai_time = (start_time - start_time1)
    # find_time = (end_time - start_time)
    # total_time = (end_time - start_time1)
    # print("Retrieve heatmap time: ", xai_time)            
    # print("Find attention points time mth1: ", find_time) 
    # print("Total time mth1: ", total_time) 
    # print("Percentage ((heatmap time)/(total time)) * 100: ", (xai_time/total_time) * 100, "\n") 
    return "list_of_ControlPointsInsideRegion", "(end_time - start_time1)", copy_orig_image

def get_attetion_region_and_darken_pixel_mth2(xai_image, orig_image, x_sqr_size, y_sqr_size, number_of_regions):
    x_dim = xai_image.shape[0]
    y_dim = xai_image.shape[1]

    # print("x_dim ",x_dim)
    # print("y_dim ",y_dim)

    greater_value_sum_xai = 0
    x_final_pos = 0
    y_final_pos = 0
    list_pos_and_values = []
    for y_sqr_pos in range(y_dim - y_sqr_size):
        for x_sqr_pos in range(x_dim - x_sqr_size):
            sum_xai = 0
            for y_in_sqr in range(y_sqr_size):
                y_pixel_pos = y_sqr_pos + y_in_sqr
                for x_in_sqr in range(x_sqr_size):                    
                    x_pixel_pos = x_sqr_pos + x_in_sqr
                    if orig_image[y_pixel_pos][x_pixel_pos] > 0:
                        sum_xai += xai_image[y_pixel_pos][x_pixel_pos]
            list_pos_and_values.append([(x_sqr_pos, y_sqr_pos), xai_image[y_pixel_pos][x_pixel_pos]])

    get_1 = itemgetter(1)
    list_pos_and_values_sorted = sorted(list_pos_and_values, key=get_1, reverse=True)
    list_pos_and_values_sorted_filtered = list_pos_and_values_sorted[0:number_of_regions]
    list_to_return = [item[0] for item in list_pos_and_values_sorted_filtered]     

    return list_to_return


def AM_darken_attention_pixels_mth2(images, x_patch_size, y_patch_size, svg_path, number_of_regions):
    """
    AM_get_attetion_svg_points_images_mth1 Iterate all the image looking for the region with more attention and return list of points (tuples) inside the square region with more attention.

    :param images: images should have the shape: (x, 28, 28) where x>=1
    :param x_patch_size: X size of the square region
    :param y_patch_size: Y size of the square region
    :param svg_path: A string with the digit's SVG path description. Ex: "M .... C .... Z".
    :return: A list of point positions that are inside the region found. A well detailed explanation about the structure of the list returned is described at the end of this function.
    """ 
    # start_time1 = time.time()
    xai = get_XAI_image(images)
    # start_time = time.time()
    # x, y = get_attetion_region(cam, images)
    list_of_ControlPointsInsideRegion = []
    for i in range(images.shape[0]):
        pattern = re.compile('([\d\.]+),([\d\.]+)\s[MCLZ]')
        ControlPoints = pattern.findall(svg_path)
        controlPoints = [(float(i[0]), float(i[1])) for i in ControlPoints]

        # x, y = get_attetion_region(xai[i], images[i], x_patch_size, y_patch_size) #Getting coordinates of the highest attetion region (patch) reference point
        list_of_regions = get_attetion_region_and_darken_pixel_mth2(xai[i], images[i], x_patch_size, y_patch_size, number_of_regions) #Getting coordinates of the highest attetion region (patch) reference point
        copy_orig_image = copy.deepcopy(images[i])
        for pos in list_of_regions:
            for x_ite in range(x_patch_size):
                for y_ite in range(y_patch_size):
                    copy_orig_image[pos[1] + y_ite][pos[0] + x_ite] = 0
        # ControlPointsInsideRegion = getControlPointsInsideAttRegion(x,y,x_patch_size,y_patch_size, controlPoints) #Getting all the points inside the highest attetion patch
        # list_of_ControlPointsInsideRegion.append(ControlPointsInsideRegion)

    # end_time = time.time()

    # xai_time = (start_time - start_time1)
    # find_time = (end_time - start_time)
    # total_time = (end_time - start_time1)
    # print("Retrieve heatmap time: ", xai_time)            
    # print("Find attention points time mth1: ", find_time) 
    # print("Total time mth1: ", total_time) 
    # print("Percentage ((heatmap time)/(total time)) * 100: ", (xai_time/total_time) * 100, "\n") 
    return "list_of_ControlPointsInsideRegion", "(end_time - start_time1)", copy_orig_image, list_of_regions, xai


def get_attetion_region(xai_image, orig_image, x_sqr_size, y_sqr_size):
    x_dim = xai_image.shape[0]
    y_dim = xai_image.shape[1]

    # print("x_dim ",x_dim)
    # print("y_dim ",y_dim)

    greater_value_sum_xai = 0
    x_final_pos = 0
    y_final_pos = 0

    for y_sqr_pos in range(y_dim - y_sqr_size):
        for x_sqr_pos in range(x_dim - x_sqr_size):
            sum_xai = 0
            for y_in_sqr in range(y_sqr_size):
                y_pixel_pos = y_sqr_pos + y_in_sqr
                for x_in_sqr in range(x_sqr_size):                    
                    x_pixel_pos = x_sqr_pos + x_in_sqr
                    # if orig_image[y_pixel_pos][x_pixel_pos] > 0:
                    sum_xai += xai_image[y_pixel_pos][x_pixel_pos]
            if sum_xai > greater_value_sum_xai:
                greater_value_sum_xai = sum_xai
                x_final_pos = x_sqr_pos
                y_final_pos = y_sqr_pos

    return x_final_pos, y_final_pos

def get_attetion_region_mth2(xai_image, orig_image, sqr_size):
    start_time = time.time()
    x_dim = xai_image.shape[0]
    y_dim = xai_image.shape[1]

    # print("x_dim ",x_dim)
    # print("y_dim ",y_dim)

    greater_value_sum_xai = 0

    if sqr_size == 3:
        y_border_up = -1
        y_border_bottom = 1
        x_border_right = 1
        x_border_left = -1

    for y_sqr_pos in range(y_dim):
        for x_sqr_pos in range(x_dim):
            if orig_image[y_sqr_pos][x_sqr_pos] > 0:
                sum_xai = 0
                for y_in_sqr in range(y_border_up, y_border_bottom + 1):
                    y_pixel_pos = y_sqr_pos + y_in_sqr
                    if y_pixel_pos >= 0 and y_pixel_pos <= y_dim - 1:
                        for x_in_sqr in range(x_border_left, x_border_right + 1):                    
                            x_pixel_pos = x_sqr_pos + x_in_sqr
                            if x_pixel_pos >= 0 and x_pixel_pos <= x_dim - 1:
                                sum_xai += xai_image[y_pixel_pos][x_pixel_pos]
                if sum_xai > greater_value_sum_xai:
                    greater_value_sum_xai = sum_xai
                    x_final_pos = x_sqr_pos
                    y_final_pos = y_sqr_pos
                    

    end_time = time.time()
    return x_final_pos, y_final_pos, (end_time - start_time)

def get_attetion_region_mth3(xai_image, svg_path_list, sqr_size):

    x_dim = xai_image.shape[0]
    y_dim = xai_image.shape[1]

    if sqr_size == 3:
        y_border_up = -1
        y_border_bottom = 1
        x_border_right = 1
        x_border_left = -1
    elif sqr_size == 5:
        y_border_up = -2
        y_border_bottom = 2
        x_border_right = 2
        x_border_left = -2
    else:
        print("Choose a valid value for square_size (sqr_size): 3 or 5")
        return 0
    
    xai_list = []
    for pos in svg_path_list:
        x_sqr_pos = int(pos[0])
        y_sqr_pos = int(pos[1])
        sum_xai = 0
        for y_in_sqr in range(y_border_up, y_border_bottom + 1):
            y_pixel_pos = y_sqr_pos + y_in_sqr
            if y_pixel_pos >= 0 and y_pixel_pos <= y_dim - 1:
                for x_in_sqr in range(x_border_left, x_border_right + 1):                    
                    x_pixel_pos = x_sqr_pos + x_in_sqr
                    if x_pixel_pos >= 0 and x_pixel_pos <= x_dim - 1:
                        sum_xai += xai_image[y_pixel_pos][x_pixel_pos]
        xai_list.append(sum_xai)

    sum_xai_list = sum(xai_list)
    # print("sum_xai_list", sum_xai_list)
    # print("xai_list len", len(xai_list))
    # print("svg_path_list len", len(svg_path_list))

    final_list =[]
    #sum_test = 0

    for sum_value, pos in zip(xai_list, svg_path_list):
        final_list.append([pos,sum_value/sum_xai_list])
        # sum_test += (sum_value/sum_xai_list) #Should be equal to 1



    # print("SUM XAI TEST:",sum_test)

    #Render XAI Images
    # f, ax = plt.subplots()
    # heatmap = np.uint8(cm.jet(xai_image) * 255)
    # ax.set_title("Time: " + str((end_time - start_time)))
    # ax.imshow(heatmap, cmap='jet')
    # ax.scatter(*zip(*svg_path_list),s=80)

    # for z, sum_value in enumerate(xai_list):
    #     ax.annotate(round((sum_value/sum_xai_list)*100,2), (svg_path_list[z][0], svg_path_list[z][1]))
    # plt.tight_layout()
    # plt.savefig("./xai/gradcam++/"+str(i)+"mth3_sqr=3_opt1.png")

    # plt.cla()

    return final_list

def get_attetion_region_mth4(xai_image, svg_path_list, sqr_size):

    start_time = time.time()
    x_dim = xai_image.shape[0]
    y_dim = xai_image.shape[1]

    if sqr_size == 3:
        y_border_up = -1
        y_border_bottom = 1
        x_border_right = 1
        x_border_left = -1
    elif sqr_size == 5:
        y_border_up = -2
        y_border_bottom = 2
        x_border_right = 2
        x_border_left = -2
    else:
        print("Choose a valid value for square_size (sqr_size): 3 or 5")
        return 0
    
    max_sum_xai = 0    
    pos_max = svg_path_list[0]
    for pos in svg_path_list:
        x_sqr_pos = int(pos[0])
        y_sqr_pos = int(pos[1])
        sum_xai = 0
        for y_in_sqr in range(y_border_up, y_border_bottom + 1):
            y_pixel_pos = y_sqr_pos + y_in_sqr
            if y_pixel_pos >= 0 and y_pixel_pos <= y_dim - 1:
                for x_in_sqr in range(x_border_left, x_border_right + 1):                    
                    x_pixel_pos = x_sqr_pos + x_in_sqr
                    if x_pixel_pos >= 0 and x_pixel_pos <= x_dim - 1:                       
                        sum_xai += xai_image[y_pixel_pos][x_pixel_pos]
        if sum_xai > max_sum_xai:
            max_sum_xai = sum_xai
            pos_max = pos
    
    # print("MAXIMUM SUM_XAI =", sum_xai)
    end_time = time.time()
    # print("SUM XAI TEST:",sum_test)

    #Render XAI Images
    # f, ax = plt.subplots()
    # heatmap = np.uint8(cm.jet(xai_image)[..., :3] * 255)
    # ax.set_title("Time: " + str((end_time - start_time)))
    # ax.imshow(heatmap, cmap='jet')
    # ax.scatter(*zip(*svg_path_list),s=80)

    # for z, sum_value in enumerate(svg_path_list):
    #     ax.annotate("("+str(svg_path_list[z][0])+","+str(svg_path_list[z][1])+")", (svg_path_list[z][0], svg_path_list[z][1]))
    # plt.tight_layout()
    # plt.savefig("./xai/"+str(time.time())+"mth3_sqr=3_opt1.png")

    # plt.cla()

    return pos_max, (end_time - start_time)

def get_attetion_region_mth5(xai_image, svg_path_list, number_of_points):

    x_dim = xai_image.shape[0]
    y_dim = xai_image.shape[1]
    
    list_pos_and_values =[]
    for pos in svg_path_list:
        x_pixel_pos = int(pos[0])
        y_pixel_pos = int(pos[1])
        if x_pixel_pos >= 0 and x_pixel_pos <= x_dim - 1:
            if y_pixel_pos >= 0 and y_pixel_pos <= y_dim - 1:
                list_pos_and_values.append([pos, xai_image[y_pixel_pos][x_pixel_pos]])
    get_1 = itemgetter(1)
    list_pos_and_values_sorted = sorted(list_pos_and_values, key=get_1, reverse=True)
    list_to_return = list_pos_and_values_sorted[0:number_of_points]
    new_list = [item[0] for item in list_to_return]
    # print("MAXIMUM SUM_XAI =", sum_xai)
    # print("SUM XAI TEST:",sum_test)

    #Render XAI Images
    # f, ax = plt.subplots()
    # heatmap = np.uint8(cm.jet(xai_image) * 255)
    # ax.set_title("Time: " + str((end_time - start_time)))
    # ax.imshow(heatmap, cmap='jet')
    # ax.scatter(*zip(*svg_path_list),s=80)

    # for z, sum_value in enumerate(xai_list):
    #     ax.annotate(round((sum_value/sum_xai_list)*100,2), (svg_path_list[z][0], svg_path_list[z][1]))
    # plt.tight_layout()
    # plt.savefig("./xai/gradcam++/"+str(i)+"mth3_sqr=3_opt1.png")

    # plt.cla()

    return [new_list]


def get_XAI_image(images):# images should have the shape: (x, 28, 28) where x>=1
    # start_time = time.time()
    images_reshaped = input_reshape_images(images)

    X = preprocess_input(images_reshaped, mode = "tf")

    # prediction = model.predict_classes(input_reshape(images))

    if Attention_Technique == "Faster-ScoreCAM":

        # Generate heatmap with Faster-ScoreCAM
        cam = scorecam(score,
                X,
                penultimate_layer=-1,
                max_N=10)
    
    elif Attention_Technique == "Gradcam++":    

        # Generate heatmap with GradCAM++
        cam = gradcam(score,
                    X,
                    penultimate_layer=-1)

    # elif Attention_Technique == "Gradcam":
    #     # Generate heatmap with GradCAM
    #     cam = gradcam(score,
    #                 X,
    #                 penultimate_layer=-1)
    # end_time = time.time()
    # print("XAI Time: ", (end_time-start_time))
    return cam

def getControlPointsInsideAttRegion(x,y,x_dim,y_dim, controlPoints):
    list_of_points = []
    foundPoint = False
    for cp in controlPoints:
        if cp[0] >= (x-1) and cp[0] < x + x_dim + 1:
            if cp[1] >= (y-1) and cp[1] < y + y_dim + 1:
                list_of_points.append(cp)
                foundPoint = True
    if foundPoint == False:
        list_of_points.append(controlPoints[0])

    return list_of_points

def AM_get_attetion_svg_points_image(image, x_patch_size, y_patch_size, model): # images should have the shape: (28, 28)
    """
    Does the same as AM_get_attetion_svg_points_images_mth1. The difference is that this function computes a single image with the format (28, 28). Use AM_get_attetion_svg_points_images_mth1. 

    :param images: images should have the shape: (x, 28, 28) where x>=1
    :param x_patch_size: X size of the square region
    :param y_patch_size: Y size of the square region
    :param model: The model object that will predict the value of the digit in the image 
    :return: A list of point positions that are inside the region found.
    """
    images = image.reshape(1, 28, 28)

    xai = get_XAI_image(images, model)
    # x, y = get_attetion_region(cam, images)
    list_of_ControlPointsInsideRegion = []
    ControlPoints = vectorization_tools.getImageControlPoints(images[0])
    x, y, elapsed_time = get_attetion_region(xai[0], images[0], x_patch_size, y_patch_size) #Getting coordinates of the highest attetion region (patch) reference point
    ControlPointsInsideRegion = getControlPointsInsideAttRegion(x,y,x_patch_size,y_patch_size, ControlPoints) #Getting all the points inside the highest attetion patch
    list_of_ControlPointsInsideRegion.append(ControlPointsInsideRegion)

    return list_of_ControlPointsInsideRegion, elapsed_time

def AM_get_attetion_svg_points_images_mth1(images, x_patch_size, y_patch_size, svg_path):
    """
    AM_get_attetion_svg_points_images_mth1 Iterate all the image looking for the region with more attention and return list of points (tuples) inside the square region with more attention.

    :param images: images should have the shape: (x, 28, 28) where x>=1
    :param x_patch_size: X size of the square region
    :param y_patch_size: Y size of the square region
    :param svg_path: A string with the digit's SVG path description. Ex: "M .... C .... Z".
    :return: A list of point positions that are inside the region found. A well detailed explanation about the structure of the list returned is described at the end of this function.
    """ 
    # start_time1 = time.time()
    xai = get_XAI_image(images)
    # start_time = time.time()
    # x, y = get_attetion_region(cam, images)
    list_of_ControlPointsInsideRegion = []
    for i in range(images.shape[0]):
        pattern = re.compile('([\d\.]+),([\d\.]+)\s[MCLZ]')
        ControlPoints = pattern.findall(svg_path)
        controlPoints = [(float(i[0]), float(i[1])) for i in ControlPoints]

        x, y = get_attetion_region(xai[i], images[i], x_patch_size, y_patch_size) #Getting coordinates of the highest attetion region (patch) reference point
        ControlPointsInsideRegion = getControlPointsInsideAttRegion(x,y,x_patch_size,y_patch_size, controlPoints) #Getting all the points inside the highest attetion patch
        list_of_ControlPointsInsideRegion.append(ControlPointsInsideRegion)

    # end_time = time.time()

    # xai_time = (start_time - start_time1)
    # find_time = (end_time - start_time)
    # total_time = (end_time - start_time1)
    # print("Retrieve heatmap time: ", xai_time)            
    # print("Find attention points time mth1: ", find_time) 
    # print("Total time mth1: ", total_time) 
    # print("Percentage ((heatmap time)/(total time)) * 100: ", (xai_time/total_time) * 100, "\n") 
    return list_of_ControlPointsInsideRegion, "(end_time - start_time1)"
    """
    #-----------------Structure of the list returned----------------#
     Start of the list -> [
                             image_0 (list)-> [                             
                                     Position (x, y) of point_0 of image_0(tuple) -> (x0,y0),                                
                                     Position (x, y) of point_1 of image_0(tuple) -> (x1,y1),
                                     (x2,y2),
                                       .
                                       .
                                       .
                                     (xn,yn)
                             ],
                             image_1 (list) -> [(x0,y0), (x1,y1), (x2,y2), (x3,y3), ... (xn,yn)],  
                             image_2 (list) -> [(x0,y0), (x1,y1), (x2,y2), (x3,y3), ... (xn,yn)],
                               .
                               .
                               .
                             image_n (list) -> [(x0,y0), (x1,y1), (x2,y2), (x3,y3), ... (xn,yn)]      
    
     End of the list -> ]
    #----------------- END Structure of the list returned----------------#
    """
def AM_get_attetion_svg_points_images_mth3(images, sqr_size, model):
    """
    AM_get_attetion_svg_points_images_mth2 Calculate the attetion score around each SVG path point and return a list of points (tuples) and the respective non-uniform distribution weights for all the SVG path points

    :param images: images should have the shape: (x, 28, 28) where x>=1
    :param sqr_size: X and Y size of the square region
    :param model: The model object that will predict the value of the digit in the image 
    :return: A a list of points (tuples) and the respective non-uniform distribution weights for all the SVG path points. A well detailed explanation about the structure of the list returned is described at the end of this function.
    """ 
    xai = get_XAI_image(images, model)
    start_time = time.time()
    # x, y = get_attetion_region(cam, images)
    list_of_points_and_weights = []

    for i in range(images.shape[0]):
        ControlPoints = vectorization_tools.getImageControlPoints(images[i])
        print("image",i )
        pos_and_prob_list = get_attetion_region_mth3(xai[i], ControlPoints, sqr_size, i)
        list_of_points_and_weights.append(pos_and_prob_list)

    end_time = time.time()            
    print("Find attention points time mth3: ", (end_time - start_time))
    return list_of_points_and_weights, (end_time - start_time)
    """
    #-----------------Structure of the list returned----------------#
     Start of the list -> [
                             image_0 (list)-> [
                                 point_0 of image_0 (list) -> [
                                     Position (x, y) of point_0 (tuple) -> (x0,y0),
                                     Weights for non-uniform distribution for point_0 (float) -> float
                                 ],
                                 point_1 of image_0 (list) -> [
                                     Position (x, y) of point_1 (tuple) -> (x1,y1),
                                     Weights for non-uniform distribution for point_1 (float) -> float
                                 ],
                                 [
                                     (x2,y2), float
                                 ],
                                 .
                                 .
                                 .
                                 [(xn,yn),float]
                             ],
                             image_1 (list) -> [[(x0,y0),float], [(x1,y1),float], [(x2,y2),float], [(x3,y3),float] ... [(xn,yn),float]],  
                             image_2 (list) -> [[(x0,y0),float], [(x1,y1),float], [(x2,y2),float], [(x3,y3),float] ... [(xn,yn),float]],
                               .
                               .
                               .
                             image_n (list) -> [[(x0,y0),float], [(x1,y1),float], [(x2,y2),float], [(x3,y3),float] ... [(xn,yn),float]]        
    
     End of the list -> ]
    #----------------- END Structure of the list returned----------------#
    """
def AM_get_attetion_svg_points_images_mth2(images, sqr_size, svg_path):
    """
    AM_get_attetion_svg_points_images_mth1 Iterate all the image looking for the region with more attention and return list of points (tuples) inside the square region with more attention.

    :param images: images should have the shape: (x, 28, 28) where x>=1 
    :param sqr_size: X and Y size of the square region
    :param svg_path: A string with the digit's SVG path description. Ex: "M .... C .... Z".
    :return: The point with more score attention around it. Tuple - One single point. Ex: (x,y)
    """ 
    xai = get_XAI_image(images)

    # x, y = get_attetion_region(cam, images)
    # list_of_ControlPointsInsideRegion = []
    total_elapsed_time = 0
    for i in range(images.shape[0]):
        pattern = re.compile('([\d\.]+),([\d\.]+)\s[MCLZ]')
        ControlPoints = pattern.findall(svg_path)
        controlPoints = [(float(i[0]), float(i[1])) for i in ControlPoints]

        position, elapsed_time = get_attetion_region_mth4(xai[i], controlPoints, sqr_size) #Getting coordinates of the highest attetion region (patch) reference point

        # list_of_ControlPointsInsideRegion.append(ControlPointsInsideRegion)
       
    return position, total_elapsed_time

def AM_get_attetion_svg_points_images_mth5(images, number_of_points, svg_path):
    """
    AM_get_attetion_svg_points_images_mth1 Iterate all the image looking for the region with more attention and return list of points (tuples) inside the square region with more attention.

    :param images: images should have the shape: (x, 28, 28) where x>=1
    :param number_of_points: Number of points (n) to return
    :param svg_path: A string with the digit's SVG path description. Ex: "M .... C .... Z".
    :return: A list of n points (number_of_points) with more score attention around it. List of tuples Ex: (x,y)
    """ 
    # start_time1= time.time() 
    xai = get_XAI_image(images)
    # start_time = time.time() 
    # x, y = get_attetion_region(cam, images)
    # list_of_ControlPointsInsideRegion = []
    total_elapsed_time = 0
    for i in range(images.shape[0]):
        pattern = re.compile('([\d\.]+),([\d\.]+)\s[MCLZ]')
        ControlPoints = pattern.findall(svg_path)
        controlPoints = [(float(i[0]), float(i[1])) for i in ControlPoints]

        # position, elapsed_time = get_attetion_region_mth4(xai[i], controlPoints, sqr_size) #Getting coordinates of the highest attetion region (patch) reference point
        position = get_attetion_region_mth5(xai[i], controlPoints, number_of_points) #Getting coordinates of the highest attetion region (patch) reference point

        # list_of_ControlPointsInsideRegion.append(ControlPointsInsideRegion)

    # end_time = time.time()
    # xai_time = (start_time - start_time1)
    # find_time = (end_time - start_time)
    # total_time = (end_time - start_time1)
    # print("Retrieve heatmap time: ", xai_time)            
    # print("Find attention points time mth5: ", find_time) 
    # print("Total time mth5: ", total_time) 
    # print("Percentage ((heatmap time)/(total time)) * 100: ", (xai_time/total_time) * 100, "\n") 
    return position, "(end_time - start_time)"

def AM_get_attetion_svg_points_images_mth6(images, number_of_points, svg_path):
    """
    AM_get_attetion_svg_points_images_mth1 Iterate all the image looking for the region with more attention and return list of points (tuples) inside the square region with more attention.

    :param images: images should have the shape: (x, 28, 28) where x>=1
    :param number_of_points: Number of points (n) to return
    :param svg_path: A string with the digit's SVG path description. Ex: "M .... C .... Z".
    :return: A list of n points (number_of_points) with more score attention around it. List of tuples Ex: (x,y)
    """ 
    # start_time1= time.time() 
    xai = get_XAI_image(images)
    # start_time = time.time() 
    # x, y = get_attetion_region(cam, images)
    # list_of_ControlPointsInsideRegion = []
    total_elapsed_time = 0
    for i in range(images.shape[0]):
        pattern = re.compile('([\d\.]+),([\d\.]+)\s[MCLZ]')
        ControlPoints = pattern.findall(svg_path)
        controlPoints = [(float(i[0]), float(i[1])) for i in ControlPoints]

        # position, elapsed_time = get_attetion_region_mth4(xai[i], controlPoints, sqr_size) #Getting coordinates of the highest attetion region (patch) reference point
        position = get_attetion_region_mth5(xai[i], controlPoints, number_of_points) #Getting coordinates of the highest attetion region (patch) reference point

        # list_of_ControlPointsInsideRegion.append(ControlPointsInsideRegion)

    # end_time = time.time()
    # xai_time = (start_time - start_time1)
    # find_time = (end_time - start_time)
    # total_time = (end_time - start_time1)
    # print("Retrieve heatmap time: ", xai_time)            
    # print("Find attention points time mth5: ", find_time) 
    # print("Total time mth5: ", total_time) 
    # print("Percentage ((heatmap time)/(total time)) * 100: ", (xai_time/total_time) * 100, "\n") 
    return position, "(end_time - start_time)", xai

def apply_displacement_to_mutant_2(list_of_points, extent):
    displ = uniform(MUTLOWERBOUND, MUTUPPERBOUND) * extent
    x_or_y = random.choice((0,1))
    y_or_x = (x_or_y-1) * -1
    list_of_mutated_coordinates_string = []
    coordinate_matutated = [0,0]
    for point in list_of_points:
        coordinate_matutated[y_or_x] = point[y_or_x]
        value = point[x_or_y]        
        if random.uniform(0, 1) >= MUTOFPROB:
            result = float(value) + displ
            coordinate_matutated[x_or_y] = result
            list_of_mutated_coordinates_string.append(str(coordinate_matutated[0])+","+str(coordinate_matutated[1]))
        else:
            result = float(value) - displ
            coordinate_matutated[x_or_y] = result
            list_of_mutated_coordinates_string.append(str(coordinate_matutated[0])+","+str(coordinate_matutated[1]))

    return list_of_mutated_coordinates_string

def apply_mutoperator_attention_2(input_img, svg_path, extent, model):
    # list_of_points_inside_square_attention_patch, elapsed_time = AM_get_attetion_svg_points_images_mth1(input_img, 3, 3,
    #                                                                                                  svg_path)
    list_of_points_inside_square_attention_patch, elapsed_time = AM_get_attetion_svg_points_images_mth5(input_img, 2, svg_path)
    
    list_of_mutated_coordinates_string = apply_displacement_to_mutant_2(list_of_points_inside_square_attention_patch[0], extent)  
    
    path = svg_path
    list_of_points = list_of_points_inside_square_attention_patch[0]                                                                                                
    for original_coordinate_tuple, mutated_coordinate_tuple in zip(list_of_points, list_of_mutated_coordinates_string):
        original_coordinate = str(original_coordinate_tuple[0]) + "," + str(original_coordinate_tuple[1])
        # print("original coordinate", original_coordinate)
        # print("mutated coordinate", mutated_coordinate_tuple)
        path = path.replace(original_coordinate, mutated_coordinate_tuple)

    return path

def apply_mutoperator_attention_2_1(input_img, svg_path, extent, square_size):
    list_of_points_inside_square_attention_patch, elapsed_time, xai = AM_get_attetion_svg_points_images_mth1_1(input_img, square_size, square_size,
                                                                                                     svg_path)
    # list_of_points_inside_square_attention_patch, elapsed_time = AM_get_attetion_svg_points_images_mth5(input_img, 2, svg_path)
    if list_of_points_inside_square_attention_patch != None:
    
        list_of_mutated_coordinates_string = apply_displacement_to_mutant_2(list_of_points_inside_square_attention_patch[0], extent)  
        
        path = svg_path
        list_of_points = list_of_points_inside_square_attention_patch[0]                                                                                                
        for original_coordinate_tuple, mutated_coordinate_tuple in zip(list_of_points, list_of_mutated_coordinates_string):
            original_coordinate = str(original_coordinate_tuple[0]) + "," + str(original_coordinate_tuple[1])
            # print("original coordinate", original_coordinate)
            # print("mutated coordinate", mutated_coordinate_tuple)
            path = path.replace(original_coordinate, mutated_coordinate_tuple)

        return path, list_of_points_inside_square_attention_patch, xai, list_of_points_inside_square_attention_patch
    else: 
        return None, None, xai
        

def AM_get_attetion_svg_points_images_mth1_1(images, x_patch_size, y_patch_size, svg_path):
    """
    AM_get_attetion_svg_points_images_mth1 Iterate all the image looking for the region with more attention and return list of points (tuples) inside the square region with more attention.

    :param images: images should have the shape: (x, 28, 28) where x>=1
    :param x_patch_size: X size of the square region
    :param y_patch_size: Y size of the square region
    :param svg_path: A string with the digit's SVG path description. Ex: "M .... C .... Z".
    :return: A list of point positions that are inside the region found. A well detailed explanation about the structure of the list returned is described at the end of this function.
    """ 
    # start_time1 = time.time()
    xai = get_XAI_image(images)
    # start_time = time.time()
    # x, y = get_attetion_region(cam, images)
    list_of_ControlPointsInsideRegion = []
    for i in range(images.shape[0]):
        pattern = re.compile('([\d\.]+),([\d\.]+)\s[MCLZ]')
        ControlPoints = pattern.findall(svg_path)
        controlPoints = [(float(i[0]), float(i[1])) for i in ControlPoints]
        if len(ControlPoints) != 0:
            x, y = get_attetion_region(xai[i], images[i], x_patch_size, y_patch_size) #Getting coordinates of the highest attetion region (patch) reference point
            ControlPointsInsideRegion = getControlPointsInsideAttRegion(x,y,x_patch_size,y_patch_size, controlPoints) #Getting all the points inside the highest attetion patch
            list_of_ControlPointsInsideRegion.append(ControlPointsInsideRegion)
        else:
            return None, "(end_time - start_time1)", xai

    # end_time = time.time()

    # xai_time = (start_time - start_time1)
    # find_time = (end_time - start_time)
    # total_time = (end_time - start_time1)
    # print("Retrieve heatmap time: ", xai_time)            
    # print("Find attention points time mth1: ", find_time) 
    # print("Total time mth1: ", total_time) 
    # print("Percentage ((heatmap time)/(total time)) * 100: ", (xai_time/total_time) * 100, "\n") 
    return list_of_ControlPointsInsideRegion, "(end_time - start_time1)", xai

def apply_displacement_to_mutant(value, extent):
    displ = uniform(MUTLOWERBOUND, MUTUPPERBOUND) * extent
    if random.uniform(0, 1) >= MUTOFPROB:
        result = float(value) + displ
    else:
        result = float(value) - displ
    return repr(result)


def L2Norm(H1,H2):
    distance =0
    for i in range(len(H1)):
        distance += np.square(H1[i]-H2[i])
    return np.sqrt(distance)

def get_distance(v1, v2):
    return np.linalg.norm(v1 - v2)

def get_svg_path(image):
    array = vectorization_tools.preprocess(image)
    # use Potrace lib to obtain a SVG path from a Bitmap
    # Create a bitmap from the array
    bmp = vectorization_tools.potrace.Bitmap(array)
    # Trace the bitmap to a path
    path = bmp.trace()
    return vectorization_tools.createSVGpath(path)

def input_reshape_images_reverse(x):
    # shape numpy vectors
    # if keras.backend.image_data_format() == 'channels_first':
    #     x_reshape = x.reshape(x.shape[0], 1, 28, 28)
    # else:
    #     x_reshape = x.reshape(x.shape[0], 28, 28, 1)
    x_reshape = x.reshape(x.shape[0], 28, 28, 1)
    x_reshape = x_reshape.astype('float32')
    x_reshape *= 255.0
    return x_reshape

def get_digit_from_MNIST(x_test, labels, lbl_required, desired_occurrence):
    current_occurrence = 0
    for i, label in enumerate(labels):
        if label == lbl_required:
            current_occurrence += 1
            if current_occurrence == desired_occurrence:
                return x_test[i]

def evaluate_ff2(predictions, lbl):
    predictions = predictions.tolist()
    prediction = predictions[0]
    confidence_expclass = prediction[lbl]
    # print("confidence_expclass", confidence_expclass)
    unexpected_prediction = prediction[0:lbl] + prediction[lbl+1:10]
    # print("unexpected_prediction", unexpected_prediction)
    confidence_unexpectclass = max(unexpected_prediction)
    fitness = confidence_expclass - confidence_unexpectclass 

    return fitness

def get_SVG_points_with_sqr_attention(xai_image, svg_path_list, sqr_size, number_of_points):

    start_time = time.time()
    x_dim = xai_image.shape[0]
    y_dim = xai_image.shape[1]

    if sqr_size == 3:
        y_border_up = -1
        y_border_bottom = 1
        x_border_right = 1
        x_border_left = -1
    elif sqr_size == 5:
        y_border_up = -2
        y_border_bottom = 2
        x_border_right = 2
        x_border_left = -2
    else:
        print("Choose a valid value for square_size (sqr_size): 3 or 5")
        return 0
    
    max_sum_xai = 0    
    # pos_max = svg_path_list[0]
    list_pos_and_values = []
    for pos in svg_path_list:
        x_sqr_pos = int(pos[0])
        y_sqr_pos = int(pos[1])
        sum_xai = 0
        for y_in_sqr in range(y_border_up, y_border_bottom + 1):
            y_pixel_pos = y_sqr_pos + y_in_sqr
            if y_pixel_pos >= 0 and y_pixel_pos <= y_dim - 1:
                for x_in_sqr in range(x_border_left, x_border_right + 1):                    
                    x_pixel_pos = x_sqr_pos + x_in_sqr
                    if x_pixel_pos >= 0 and x_pixel_pos <= x_dim - 1:                       
                        sum_xai += xai_image[y_pixel_pos][x_pixel_pos]
        list_pos_and_values.append([pos, sum_xai])                
        if sum_xai > max_sum_xai:
            max_sum_xai = sum_xai
            pos_max = pos
    
    
    get_1 = itemgetter(1)
    list_pos_and_values_sorted = sorted(list_pos_and_values, key=get_1, reverse=True)
    list_to_return = list_pos_and_values_sorted[0:number_of_points]
    new_list = [item[0] for item in list_to_return]
    # print("MAXIMUM SUM_XAI =", sum_xai)
    end_time = time.time()
    # print("SUM XAI TEST:",sum_test)

    #Render XAI Images
    # f, ax = plt.subplots()
    # heatmap = np.uint8(cm.jet(xai_image)[..., :3] * 255)
    # ax.set_title("Time: " + str((end_time - start_time)))
    # ax.imshow(heatmap, cmap='jet')
    # ax.scatter(*zip(*svg_path_list),s=80)

    # for z, sum_value in enumerate(svg_path_list):
    #     ax.annotate("("+str(svg_path_list[z][0])+","+str(svg_path_list[z][1])+")", (svg_path_list[z][0], svg_path_list[z][1]))
    # plt.tight_layout()
    # plt.savefig("./xai/"+str(time.time())+"mth3_sqr=3_opt1.png")

    # plt.cla()

    return [new_list] #, (end_time - start_time)

def AM_get_attetion_svg_points_images_v1(images, number_of_points, svg_path, sqr_size):
    """
    AM_get_attetion_svg_points_images_mth1 Iterate all the image looking for the region with more attention and return list of points (tuples) inside the square region with more attention.

    :param images: images should have the shape: (x, 28, 28) where x>=1
    :param number_of_points: Number of points (n) to return
    :param svg_path: A string with the digit's SVG path description. Ex: "M .... C .... Z".
    :return: A list of n points (number_of_points) with more score attention around it. List of tuples Ex: (x,y)
    """ 
    # start_time1= time.time() 
    xai = get_XAI_image(images)
    # start_time = time.time() 
    # x, y = get_attetion_region(cam, images)
    # list_of_ControlPointsInsideRegion = []
    total_elapsed_time = 0
    for i in range(images.shape[0]):
        pattern = re.compile('([\d\.]+),([\d\.]+)\s[MCLZ]')
        ControlPoints = pattern.findall(svg_path)
        controlPoints = [(float(i[0]), float(i[1])) for i in ControlPoints]

        # position, elapsed_time = get_attetion_region_mth4(xai[i], controlPoints, sqr_size) #Getting coordinates of the highest attetion region (patch) reference point
        positions = get_SVG_points_with_sqr_attention(xai[i], controlPoints, sqr_size, number_of_points) #Getting coordinates of the highest attetion region (patch) reference point
        # print("positions", positions)

        # list_of_ControlPointsInsideRegion.append(ControlPointsInsideRegion)

    # end_time = time.time()
    # xai_time = (start_time - start_time1)
    # find_time = (end_time - start_time)
    # total_time = (end_time - start_time1)
    # print("Retrieve heatmap time: ", xai_time)            
    # print("Find attention points time mth5: ", find_time) 
    # print("Total time mth5: ", total_time) 
    # print("Percentage ((heatmap time)/(total time)) * 100: ", (xai_time/total_time) * 100, "\n") 
    return positions, "(end_time - start_time)", xai

def apply_mutoperator_attention(input_img, svg_path, extent, square_size, number_of_points):
    list_of_svg_points, elapsed_time, xai = AM_get_attetion_svg_points_images_v1(input_img, number_of_points,
                                                                                                        svg_path, square_size)
    # list_of_points_inside_square_attention_patch, elapsed_time = AM_get_attetion_svg_points_images_mth5(input_img, 2, svg_path)
    # print("list_of_svg_points", list_of_svg_points)
    if len(list_of_svg_points[0]) != 0:
        original_point = random.choice(list_of_svg_points[0])
        original_coordinate = random.choice(original_point)

        mutated_coordinate = apply_displacement_to_mutant(original_coordinate, extent)


        path = svg_path.replace(str(original_coordinate), str(mutated_coordinate))

        # TODO: it seems that the points inside the square attention patch do not precisely match the point coordinates in the svg, to be tested
        return path, list_of_svg_points, xai, original_point
    else:
        return svg_path, None, xai, None

def apply_mutoperator1(input_img, svg_path, extent):

    while(True):
        # find all the vertexes
        pattern = re.compile('([\d\.]+),([\d\.]+)\s[MCLZ]')
        segments = pattern.findall(svg_path)
        # print("segments", segments)

        svg_iter = re.finditer(pattern, svg_path)
        # print("svg_iter", svg_iter)
        # chose a random vertex
        num_matches = len(segments) * 2

        random_coordinate_index = randint(0, num_matches - 1)
        # print(random_coordinate_index)

        vertex = next(value for index, value in enumerate(svg_iter) if int(index == int(random_coordinate_index / 2)))
        group_index = (random_coordinate_index % 2) + 1

        value = apply_displacement_to_mutant(vertex.group(group_index), extent)

        if 0 <= float(value) <= 28:
            break

    path = svg_path[:vertex.start(group_index)] + value + svg_path[vertex.end(group_index):]
    return path, vertex.group(group_index)

def apply_mutoperator_custom(input_img, svg_path, extent):

    while(True):
        # find all the vertexes
        pattern = re.compile('([\d\.]+),([\d\.]+)\s[MCLZ]')
        segments = pattern.findall(svg_path)
        # print("segments", segments)
        segments = random.choices(segments, k=5)

        svg_iter = re.finditer(pattern, svg_path)
        # print("svg_iter", svg_iter)
        # chose a random vertex
        num_matches = len(segments) * 2

        random_coordinate_index = randint(0, num_matches - 1)
        # print(random_coordinate_index)

        vertex = next(value for index, value in enumerate(svg_iter) if int(index == int(random_coordinate_index / 2)))
        group_index = (random_coordinate_index % 2) + 1

        value = apply_displacement_to_mutant(vertex.group(group_index), extent)

        if 0 <= float(value) <= 28:
            break

    path = svg_path[:vertex.start(group_index)] + value + svg_path[vertex.end(group_index):]
    return path, vertex.group(group_index)


def apply_mutoperator2(input_img, svg_path, extent):
    # find all the vertexes
    pattern = re.compile('C\s([\d\.]+),([\d\.]+)\s([\d\.]+),([\d\.]+)\s')
    segments = pattern.findall(svg_path)

    # chose a random control point
    num_matches = len(segments) * 4
    path = svg_path
    if num_matches > 0:
        random_coordinate_index = randint(0, num_matches - 1)
        svg_iter = re.finditer(pattern, svg_path)
        control_point = next(value for index, value in enumerate(svg_iter) if int(index == int(random_coordinate_index/4)))
        # print("control_point", control_point)
        group_index = (random_coordinate_index % 4) + 1
        # print("control_point.group(group_index)", control_point.group(group_index))
        # print("group_index", group_index)
        value = apply_displacement_to_mutant(control_point.group(group_index), extent)
        path = svg_path[:control_point.start(group_index)] + value + svg_path[control_point.end(group_index):]
    else:
        print("ERROR")
        print(svg_path)
    return path, control_point

def generate_mutant(image, extent, square_size, number_of_points, mutation_method, ATTENTION_METHOD):

    # image = copy.deepcopy(image_orig)
    # list_of_points_inside_square_attention_patch, elapsed_time = AM_get_attetion_svg_points_images_mth1(image, square_size, square_size, get_svg_path(input[0]))
    if mutation_method == True:
        if ATTENTION_METHOD == "mth5":
            mutante_digit_path, list_of_svg_points, xai, point_mutated = apply_mutoperator_attention(image, get_svg_path(image[0]), extent, square_size, number_of_points)
        elif ATTENTION_METHOD == "mth1":
            mutante_digit_path, list_of_svg_points, xai, point_mutated = apply_mutoperator_attention_2_1(image, get_svg_path(image[0]), extent, square_size)
        # print(mutante_digit_path)
        if list_of_svg_points != None and ("C" in mutante_digit_path) and ("M" in mutante_digit_path):
            rast_nparray = rasterization_tools.rasterize_in_memory(vectorization_tools.create_svg_xml(mutante_digit_path))
            # print("original_mutated_digit shape", rast_nparray.shape)
            # print("original_mutated_digit max", rast_nparray.max())
            # print("original_mutated_digit min", rast_nparray.min())   
            return rast_nparray, list_of_svg_points, xai, point_mutated
        else:
            return image, None, xai, None
    else: 
        mutante_digit_path, point_mutated = apply_mutoperator1(image, get_svg_path(image[0]), extent)
        rast_nparray = rasterization_tools.rasterize_in_memory(vectorization_tools.create_svg_xml(mutante_digit_path))
        # print("original_mutated_digit shape", rast_nparray.shape)
        # print("original_mutated_digit max", rast_nparray.max())
        # print("original_mutated_digit min", rast_nparray.min())  
        return rast_nparray, point_mutated    

def save_image(mutant_image_normal, mutant_image_att, xai_image, list_of_svg_points, iteration_list, fitness_function_att, prediction_function_att, fitness_function_normal, prediction_function_normal, number_of_mutations, folder_path, pred_normal, pred_att, ATTENTION_METHOD, square_size, iteration):
    fig = plt.figure(figsize=(9,10))
    gs = gridspec.GridSpec(nrows=3,ncols=3, width_ratios=[1,1,1], height_ratios=[1,1,1])
    ax0 = fig.add_subplot(gs[0,0])
    ax0.imshow(mutant_image_normal.reshape(28, 28), cmap = "gray")
    ax0.set_title("Normal Mutation/Pred = " + str(pred_normal), color="red")

    ax1 = fig.add_subplot(gs[0,1])
    ax1.imshow(mutant_image_att.reshape(28, 28), cmap = "gray")
    ax1.set_title("Attention Mutation/Pred= " + str(pred_att), color="blue")

    ax2 = fig.add_subplot(gs[0,2])
    ax2.imshow(xai_image[0], cmap = "jet")
    # ax2.imshow(image.reshape(28,28), cmap = "gray")
    ax2.scatter(*zip(*list_of_svg_points[0]))
    # print("list_of_regions", list_of_svg_points[0])
    # for region_pos in list_of_regions[0]:
    #     rect = patches.Rectangle((region_pos[0], region_pos[1]), square_size, square_size, linewidth=1, edgecolor='r', facecolor='none')
    #     ax[2].add_patch(rect)
    # plt.tight_layout()
    if ATTENTION_METHOD == "mth5":
        for pos in list_of_svg_points[0]:
            x = pos[0]
            y = pos[1]
            x_rounded = round(x,2)
            y_rounded = round(y,2)
            ax2.annotate("("+str(x_rounded)+","+ str(y_rounded)+")", (pos[0], pos[1]))
            rect = patches.Rectangle((x-(square_size/2), y-(square_size/2)), square_size, square_size, linewidth=1, edgecolor='r', facecolor='none')
            ax2.add_patch(rect)
    elif ATTENTION_METHOD == "mth1":
        for pos in list_of_svg_points[0]:
            x = pos[0]
            y = pos[1]
            x_rounded = round(x,2)
            y_rounded = round(y,2)
            ax2.annotate("("+str(x_rounded)+","+ str(y_rounded)+")", (pos[0], pos[1]))
    ax_fitness = fig.add_subplot(gs[1,:])
    ax_fitness.plot(iteration_list, fitness_function_att, "b", label = "Attention Algorithm")
    ax_fitness.plot(iteration_list[-1], fitness_function_att[-1], marker="o", markeredgecolor = "blue")
    ax_fitness.plot(iteration_list, fitness_function_normal, "r", label = "Normal Algorithm")
    ax_fitness.plot(iteration_list[-1], fitness_function_normal[-1], marker="o", markeredgecolor = "blue")
    ax_fitness.set_title("Fitness ff2 vs Iteration")
    ax_fitness.set_xlabel("Iteration")
    ax_fitness.set_ylabel("Fitness ff2")
    # ax_fitness.set_xticks([10,20,30,40,50,60,70,80,90,100])
    # ax_fitness.set_xlim([0, number_of_mutations])
    ax_fitness.set_ylim([0.1, 1.1])
    # ax_fitness.set_yticks([0.5,0.6,0.7,0.8,0.9,1])
    ax_fitness.grid(True)

    # print("prediction_function", prediction_function)
    ax_predictions = fig.add_subplot(gs[2,:])
    ax_predictions.plot(iteration_list, prediction_function_att, "b" ,label = "Attention Algorithm")
    ax_predictions.plot(iteration_list[-1], prediction_function_att[-1], marker="o", markeredgecolor = "blue")
    ax_predictions.plot(iteration_list, prediction_function_normal, "r" ,label = "Normal Algorithm")
    ax_predictions.plot(iteration_list[-1], prediction_function_normal[-1], marker="o", markeredgecolor = "blue")
    ax_predictions.set_title("Mutant Prediction Probablity vs Iteration")
    ax_predictions.set_xlabel("Iteration")
    ax_predictions.set_ylabel("Mutant Prediction Probablity")
    # ax_predictions.set_xlim([0, number_of_mutations])
    ax_predictions.set_ylim([0.1, 1.1])
    ax_predictions.grid(True)
    

    plt.tight_layout()
    plt.savefig(folder_path + "/" + str(iteration) + "_predATR=" + str(pred_att) + "_predNOR=" + str(pred_normal))
    plt.cla()

def save_images(mutant_image_normal_list, mutant_image_att_list, xai_image_list, list_of_svg_points_list, iteration_list, fitness_function_att, prediction_function_att, fitness_function_normal, prediction_function_normal, number_of_mutations, folder_path, pred_normal_list, pred_att_list, ATTENTION_METHOD, square_size):
    if(len(iteration_list)) > 20:
        print_interval = int(len(iteration_list)/20)
    else:
        print_interval = 1
    for img_index in range(len(iteration_list)):
        if img_index % print_interval == 0 or img_index == len(iteration_list) - 1:
            fig = plt.figure(figsize=(9,10))
            gs = gridspec.GridSpec(nrows=3,ncols=3, width_ratios=[1,1,1], height_ratios=[1,1,1])
            ax0 = fig.add_subplot(gs[0,0])
            ax0.imshow(mutant_image_normal_list[img_index].reshape(28, 28), cmap = "gray")
            ax0.set_title("Normal Mutation/Pred = " + str(pred_normal_list[img_index]), color="red")

            ax1 = fig.add_subplot(gs[0,1])
            ax1.imshow(mutant_image_att_list[img_index].reshape(28, 28), cmap = "gray")
            ax1.set_title("Attention Mutation/Pred= " + str(pred_att_list[img_index]), color="blue")

            ax2 = fig.add_subplot(gs[0,2])
            ax2.imshow(xai_image_list[img_index][0], cmap = "jet")
            # print("SVG_points:", list_of_svg_points_list[img_index][0])
            ax2.scatter(*zip(*list_of_svg_points_list[img_index][0]))
            # print("list_of_regions", list_of_svg_points[0])
            # for region_pos in list_of_regions[0]:
            #     rect = patches.Rectangle((region_pos[0], region_pos[1]), square_size, square_size, linewidth=1, edgecolor='r', facecolor='none')
            #     ax[2].add_patch(rect)
            # plt.tight_layout()
            if ATTENTION_METHOD == "mth5":
                for pos in list_of_svg_points_list[img_index][0]:
                    x = pos[0]
                    y = pos[1]
                    x_rounded = round(x,2)
                    y_rounded = round(y,2)
                    ax2.annotate("("+str(x_rounded)+","+ str(y_rounded)+")", (pos[0], pos[1]))
                    rect = patches.Rectangle((x-(square_size/2), y-(square_size/2)), square_size, square_size, linewidth=1, edgecolor='r', facecolor='none')
                    ax2.add_patch(rect)
            elif ATTENTION_METHOD == "mth1":
                for pos in list_of_svg_points_list[img_index][0]:
                    x = pos[0]
                    y = pos[1]
                    x_rounded = round(x,2)
                    y_rounded = round(y,2)
                    ax2.annotate("("+str(x_rounded)+","+ str(y_rounded)+")", (pos[0], pos[1]))
            ax_fitness = fig.add_subplot(gs[1,:])
            ax_fitness.plot(iteration_list, fitness_function_att, "b", label = "Attention Algorithm")
            ax_fitness.plot(iteration_list[img_index], fitness_function_att[img_index], marker="o", markeredgecolor = "blue")
            ax_fitness.plot(iteration_list, fitness_function_normal, "r", label = "Normal Algorithm")
            ax_fitness.plot(iteration_list[img_index], fitness_function_normal[img_index], marker="o", markeredgecolor = "blue")
            ax_fitness.set_title("Fitness ff2 vs Iteration")
            ax_fitness.set_xlabel("Iteration")
            ax_fitness.set_ylabel("Fitness ff2")
            # ax_fitness.set_xticks([10,20,30,40,50,60,70,80,90,100])
            # ax_fitness.set_xlim([0, number_of_mutations])
            ax_fitness.set_ylim([0.1, 1.1])
            # ax_fitness.set_yticks([0.5,0.6,0.7,0.8,0.9,1])
            ax_fitness.grid(True)

            # print("prediction_function", prediction_function)
            ax_predictions = fig.add_subplot(gs[2,:])
            ax_predictions.plot(iteration_list, prediction_function_att, "b" ,label = "Attention Algorithm")
            ax_predictions.plot(iteration_list[img_index], prediction_function_att[img_index], marker="o", markeredgecolor = "blue")
            ax_predictions.plot(iteration_list, prediction_function_normal, "r" ,label = "Normal Algorithm")
            ax_predictions.plot(iteration_list[img_index], prediction_function_normal[img_index], marker="o", markeredgecolor = "blue")
            ax_predictions.set_title("Mutant Prediction Probablity vs Iteration")
            ax_predictions.set_xlabel("Iteration")
            ax_predictions.set_ylabel("Mutant Prediction Probablity")
            # ax_predictions.set_xlim([0, number_of_mutations])
            ax_predictions.set_ylim([0.1, 1.1])
            ax_predictions.grid(True)
            

            plt.tight_layout()
            plt.savefig(folder_path + "/iteration=" + str(img_index) + "_predATR=" + str(pred_att_list[img_index]) + "_predNOR=" + str(pred_normal_list[img_index]))
            plt.cla()

def create_folder(mutant_root_folder, number_of_mutations, repetition, extent, label, image_index, method, attention, run_id):
    # run_id_2 = str(Timer.start.strftime('%s'))
    # DST = "mutants/debug/debug_"+ Mth1_str + run_id +"/NM="+ str(number_of_mutations) + "_REP=" + str(repetition) + "_ext="+str(extent)+"_lbl="+str(label)+"_IMG_INDEX="+str(image_index)+"_mth="+method+"_ATT="+str(attention)#+"_run_"+str(run_id_2)
    # DST = mutant_root_folder +"/NM="+ str(number_of_mutations) + "_REP=" + str(repetition) + "_ext="+str(extent)+"_lbl="+str(label)+"_IMG_INDEX="+str(image_index)+"_mth="+method+"_ATT="+str(attention)#+"_run_"+str(run_id_2)
    DST = mutant_root_folder + "/IMG_INDEX="+str(image_index) + "_REP=" + str(repetition) + "_lbl="+str(label)
    if not exists(DST):
        makedirs(DST)

    return DST
    # DST_ARC = join(DST, "archive")
    # DST_IND = join(DST, "inds")

def make_gif(frame_folder ,gif_path):
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.png")]
    if len(frames) != 0:
        frame_one = frames[0]
        frame_one.save(gif_path + ".gif", format="GIF", append_images=frames,
        save_all=True, duration=100, loop=0)

def input_reshape_images_reverse_orig(x):
    # shape numpy vectors
    if keras.backend.image_data_format() == 'channels_first':
        x_reshape = x.reshape(x.shape[0], 1, 28, 28)
    else:
        x_reshape = x.reshape(x.shape[0], 28, 28, 1)
    x_reshape = x_reshape.astype('float32')
    x_reshape *= 255.0
    return x_reshape

def normalize_2d(matrix):
    norm = np.linalg.norm(matrix)
    matrix = matrix/norm  # normalized matrix
    return matrix

def option1():
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    model = keras.models.load_model(MODEL)

    ATT_METH = False
    n = 1000
    images = x_test[:n]
    labels = y_test[:n]
    extent = 10
    square_size = 3
    print("MNIST images shape", images.shape)
    print("Method1:\n")
    for image_index in range(images.shape[0]):
        print("Image ", str(image_index))
        image = images[image_index].reshape(1,28,28)
        label = labels[image_index]
        for iteration in range(0,19):
            print("Iteration ", str(iteration))
            # list_of_points_inside_square_attention_patch, elapsed_time, rast_nparray = AM_darken_attention_pixels_mth1(image, square_size, square_size, get_svg_path(image[0]))
            # list_of_points_inside_square_attention_patch, elapsed_time, rast_nparray, list_of_regions, xai_image = AM_darken_attention_pixels_mth2(image, square_size, square_size, get_svg_path(image[0]), 6)
            # prediction = model.predict_classes(input_reshape_images(rast_nparray.reshape(1,28,28)))
            list_of_points_inside_square_attention_patch, elapsed_time = AM_get_attetion_svg_points_images_mth1(image, square_size, square_size, get_svg_path(image[0]))
            if ATT_METH == True:
                mutante_digit_path = apply_mutoperator_attention_2(image, get_svg_path(image[0]), extent, model)
            else:
                mutante_digit_path = apply_mutoperator1(image, get_svg_path(image[0]), extent)
            rast_nparray = rasterization_tools.rasterize_in_memory(vectorization_tools.create_svg_xml(mutante_digit_path))    
            prediction = model.predict_classes(input_reshape_images_reverse_orig(rast_nparray))
            prediction_mnist_data = model.predict_classes(input_reshape_images(image))
            print("PM: ",str(prediction[0]), " PO:", str(prediction_mnist_data[0]), "Label: ", str(label))        
            if prediction!= prediction_mnist_data:
            # if True:
                # dist = get_distance(input_reshape_images_reverse(rast_nparray), image)
                dist = round(get_distance(rast_nparray.reshape(1,28,28), image), 2)
                f, ax = plt.subplots(ncols = 3)
                ax[1].imshow(rast_nparray.reshape(28, 28), cmap = "gray")
                ax[1].set_title("Prediction= " + str(prediction[0]) + "\n" + "/Dist =" + str(dist))
                ax[0].imshow(image[0], cmap = "gray")
                ax[0].set_title("Prediction= " + str(prediction_mnist_data[0]))
                # ax[2].imshow(xai_image[0], cmap = "jet")
                # for region_pos in list_of_regions:
                #     rect = patches.Rectangle((region_pos[0], region_pos[1]), square_size, square_size, linewidth=1, edgecolor='r', facecolor='none')
                #     ax[2].add_patch(rect)
                # plt.tight_layout()
                plt.savefig("./mutants/mutant_img="+str(image_index)+"_ext="+str(extent)+"_sqr="+str(square_size)+"_Pred="+str(prediction[0])+"_PredOrig="+str(prediction_mnist_data[0])+"_lab="+str(label)+"_"+str(iteration)+'.png')
                plt.cla()

def option2():
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    model = keras.models.load_model(MODEL)

    LABEL = 5
    METHOD = "remut"
    # METHOD_LIST = ["remut","NOremut"]
    METHOD_LIST = ["NOremut"]#,"remut"]
    ATT_METH = True
    ATT_METH_LIST = [True, False]
    OCCURENCE_LIST = list(range (1,5))
    LABEL_LIST = list(range (0,10))
    n = 1000
    extent = 10
    number_of_points = 6
    square_size = 5
    images = x_test[:n]
    labels = y_test[:n]
    number_of_mutations = 20
    OCCURENCE = 2  
    MTH1 = True
    Mth1_str = ""
    if MTH1 == True: 
        Mth1_str = "mth1"
    run_id = str(Timer.start.strftime('%s')) 
    DST = "mutants/debug/debug_" + Mth1_str + run_id
    makedirs(DST)
    csv_path = DST + "/stats.csv"
    if os.path.exists(csv_path):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not

    with open(csv_path, append_write) as f1:
        writer = csv.writer(f1)
        writer.writerow(["Method", "Label", "Occurece", "Att_Meth", "Iteration", "Prediction"])
    
    for OCCURENCE in OCCURENCE_LIST:
        print("OCCURENCE: ", OCCURENCE)  
        for LABEL in LABEL_LIST:
            print("LABEL: ", LABEL)        
            for METHOD in METHOD_LIST:
                print("METHOD: ", METHOD)          
                for ATT_METH in ATT_METH_LIST:
                    print("ATT_METH: ", ATT_METH)
                    digit = copy.deepcopy(get_digit_from_MNIST(x_test, labels, LABEL, OCCURENCE).reshape(1,28,28))
                    original_digit = get_digit_from_MNIST(x_test, labels, LABEL, OCCURENCE).reshape(1,28,28)
                    # print("original_digit shape", original_digit.shape)
                    # print("original_digit max", original_digit.max())
                    # print("original_digit min", original_digit.min())
                    xai = get_XAI_image(original_digit)
                    iteration = 0
                    pred_input_mutant = model.predict_classes(input_reshape_images(digit))
                    digit_reshaped = input_reshape_images(digit)
                    iteration_list = []
                    fitness_function = []
                    prediction_function = []
                    # folder_path = create_folder(DST, number_of_mutations, number_of_points, extent, LABEL, OCCURENCE, METHOD, ATT_METH, run_id)
                    # print("Folder Path Created", folder_path)
                    while (iteration < number_of_mutations):
                        iteration += 1
                        # xai = get_XAI_image(input_reshape_images_reverse(digit_reshaped))
                        # print("digit shape", digit_reshaped.shape)
                        # print("digit max", digit_reshaped.max())
                        # print("digit min", digit_reshaped.min())
                        pred_input = model.predict_classes(digit_reshaped)
                        # print("pred_input", pred_input)
                        pred_class = model.predict(input_reshape_images(digit))
                        # print("pred_class", pred_class)
                        fitness = evaluate_ff2(pred_class, LABEL)
                        # print("fitness", fitness)
                        if ATT_METH == True:
                            mutant_digit, list_of_svg_points, xai = generate_mutant(input_reshape_images_reverse(digit_reshaped), extent, square_size, number_of_points, ATT_METH) 
                            print(list_of_svg_points)
                            if list_of_svg_points == None: 
                                break
                                print("aqui1")
                            f, ax = plt.subplots(ncols = 3)
                        else:
                            mutant_digit = generate_mutant(input_reshape_images_reverse(digit_reshaped), extent, square_size, number_of_points, ATT_METH)
                            f, ax = plt.subplots(ncols = 2) 
                        # mutant_digit_reshaped = input_reshape_images_reverse(mutant_digit)
                        # if type(mutant_digit) != None:
                            # print("mutant_digit shape", mutant_digit.shape)
                            # print("mutant_digit max", mutant_digit.max())
                            # print("mutant_digit min", mutant_digit.min())       
                        pred_input_mutant = model.predict_classes(mutant_digit)
                        # print("pred_input_mutant", pred_input_mutant)
                        pred_class_mutant = model.predict(mutant_digit)
                        # print("pred_class_mutant", pred_class_mutant)
                        fitness_mutant = evaluate_ff2(pred_class_mutant, LABEL)
                        
                        
                        # print("fitness_mutant", fitness_mutant)
                        iteration_list.append(iteration)
                        fitness_function.append(fitness_mutant)
                        prediction_function.append(pred_class_mutant[0][LABEL])
                        if pred_input_mutant[0] != LABEL: 
                            save_image(digit_reshaped, mutant_digit, xai, list_of_svg_points, iteration_list, fitness_function, prediction_function, number_of_mutations, DST, pred_input_mutant[0])
                            with open(csv_path, "a") as f1:
                                writer = csv.writer(f1)
                                writer.writerow([METHOD, LABEL, OCCURENCE, ATT_METH, iteration, pred_class_mutant[0][LABEL]])
                            break
                        if fitness_mutant < fitness or (iteration % 10) == 0:
                            print(iteration)

                            # iteration_list.append(iteration)
                            # fitness_function.append(fitness_mutant)
                            if ATT_METH == False: list_of_svg_points = None
                            # if (iteration % 10) == 0:
                                # save_image(digit_reshaped, mutant_digit, xai, list_of_svg_points, iteration_list, fitness_function, prediction_function, number_of_mutations, folder_path, pred_input_mutant[0])
                            
                            if (fitness_mutant < fitness): 
                                # print("FITNESS!!!!")
                                if METHOD == "remut":
                                    digit_reshaped = mutant_digit

def option3():

    from config import MUTANTS_ROOT_FOLDER,\
        METHOD_LIST,\
        ATTENTION_METHOD,\
        SAVE_IMAGES,\
        N,\
        EXTENT,\
        NUMBER_OF_POINTS,\
        SQUARE_SIZE,\
        NUMBER_OF_MUTATIONS,\
        NUMBER_OF_REPETITIONS,\
        RANDOM_SEED

    random.seed(RANDOM_SEED)
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    model = keras.models.load_model(MODEL)

    MUTANTS_ROOT_FOLDER = MUTANTS_ROOT_FOLDER  
    METHOD_LIST = METHOD_LIST
    ATTENTION_METHOD = ATTENTION_METHOD
    SAVE_IMAGES = SAVE_IMAGES
    n = N
    extent = EXTENT
    number_of_points = NUMBER_OF_POINTS
    square_size = SQUARE_SIZE
    images = x_test[44:n]
    labels = y_test[44:n]
    #Do a shuffle
    # mylist = ["apple", "banana", "cherry"]
    # random.shuffle(mylist)  

    indices = np.arange(images.shape[0])
    np.random.shuffle(indices)

    images = images[indices]
    labels = labels[indices]
    # a, array([3, 4, 1, 2, 0])
    # b, array([8, 9, 6, 7, 5])

    number_of_mutations = NUMBER_OF_MUTATIONS
    number_of_repetitions = NUMBER_OF_REPETITIONS

    # Creating CSVs in the MUTANTS_ROOT_FOLDER
    run_id = str(Timer.start.strftime('%s')) 
    DST = MUTANTS_ROOT_FOLDER + "debug_" + ATTENTION_METHOD + "_NM=" + str(number_of_mutations) + "_NR=" + str(number_of_repetitions) + "_EXT=" + str(extent) + "_NP=" + str(number_of_points) + "_SQRS="+ str(square_size) + "_MutType=" + METHOD_LIST[0] + "_ID=" + run_id
    makedirs(DST)
    csv_path = DST + "/stats.csv"
    if os.path.exists(csv_path):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not

    with open(csv_path, append_write) as f1:
        writer = csv.writer(f1)
        writer.writerow(["IMG_Index", "Algorithm", "Mut_Method", "Label", "Prediction", "Probability", "Iteration"]) 

    csv_path_2 = DST + "/stats_2.csv"
    if os.path.exists(csv_path_2):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not

    with open(csv_path_2, append_write) as f1:
        writer = csv.writer(f1)
        writer.writerow(["IMG_Index", "Label", "Repetition", "#Iterations_Att", "#Iterations_Normal", "Winner Method"]) 

    csv_path_3 = DST + "/stats_3.csv"
    if os.path.exists(csv_path_3):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not

    with open(csv_path_3, append_write) as f1:
        writer = csv.writer(f1)
        writer.writerow(["IMG_Index", "Label", "Its_Mean_Att", "Its_Mean_Normal", "Its_Std_Att", "Its_Std_Normal", "#MissClass_found_att","#MissClass_found_Normal"])

    csv_path_4 = DST + "/stats_4.csv"
    if os.path.exists(csv_path_3):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not

    with open(csv_path_4, append_write) as f1:
        writer = csv.writer(f1)
        writer.writerow(["Iteration", "Point Mutated Att","Point Mutated Normal", "List of points to be mutated Att"])

    start_time = time.time()    
    for METHOD in METHOD_LIST:
        print("METHOD: ", METHOD) 
                             
        for image_index in range(images.shape[0]):
            print("Image ", str(image_index))
            image = images[image_index].reshape(1,28,28)
            label = labels[image_index]
            LABEL = label
            digit_1 = copy.deepcopy(image)
            digit_2 = copy.deepcopy(image)
            
            iteration = 0
            
            iterations_detection_normal_list = []
            iterations_detection_att_list = []
            for REPETITION in range(1,number_of_repetitions + 1):
                print("Repetition", REPETITION)
                digit_reshaped_1 = input_reshape_images(digit_1)
                digit_reshaped_2 = input_reshape_images(digit_2)
                iteration_list = []
                fitness_function_att = []
                prediction_function_att = []
                fitness_function_normal = []
                prediction_function_normal = []
                mutant_digit_att_list = []
                mutant_digit_normal_list =[]
                xai_images_list = []
                list_of_svg_points_list = []
                pred_input_mutant_att_list = []
                pred_input_mutant_normal_list = []
                mutated_points_att_list = []
                mutated_points_normal_list = []
                list_of_svg_points_list_2 = []
                if ATTENTION_METHOD == "mth1": number_of_points = "NA"
                miss_classification_found_att = False
                miss_classification_found_normal = False
                method_winner = None
                iterations_detection_att = "NA"
                iterations_detection_normal = "NA"
                iteration = 0
                while (iteration < number_of_mutations):
                    iteration += 1                    
                    print("Iteration", iteration)

                    # Check if a missclassification was already found for ATTENTION Method. If yes, the digit will not be mutated anymore
                    if miss_classification_found_att == False:
                    # if False or iteration < 2:
                        pred_class_1 = model.predict(digit_reshaped_1)
                        fitness_1 = evaluate_ff2(pred_class_1, LABEL)                                                
                        mutant_digit_att, list_of_svg_points, xai, point_mutated = generate_mutant(input_reshape_images_reverse(digit_reshaped_1), extent, square_size, number_of_points, True, ATTENTION_METHOD) 

                        #If there is no highest attention point found, it means the digit mutated is close to an invalid digit and we can stop mutating using Attention Method
                        if list_of_svg_points == None: 
                            break    
                        list_of_svg_points_2 = list_of_svg_points 
                        if ATTENTION_METHOD == "mth5":
                            list_of_svg_points = [[point_mutated]]    
                        shape = mutant_digit_att.shape
                        pred_input_mutant_att = model.predict_classes(mutant_digit_att)
                        pred_class_mutant_att = model.predict(mutant_digit_att)
                        fitness_mutant_att = evaluate_ff2(pred_class_mutant_att, LABEL)
                        mutated_points_att_list.append(point_mutated)
                    else:
                        mutated_points_att_list.append("NA")

                    # Check if a missclassification was already found for NORMAL Method. If yes, the digit will not be mutated anymore
                    if miss_classification_found_normal == False:
                    # if False or iteration < 2:
                        pred_class_2 = model.predict(digit_reshaped_2)
                        fitness_2 = evaluate_ff2(pred_class_2, LABEL)
                        mutant_digit_normal, point_mutated_normal = generate_mutant(input_reshape_images_reverse(digit_reshaped_2), extent, square_size, number_of_points, False, ATTENTION_METHOD)
                        pred_input_mutant_normal = model.predict_classes(mutant_digit_normal)
                        pred_class_mutant_normal = model.predict(mutant_digit_normal)
                        fitness_mutant_normal = evaluate_ff2(pred_class_mutant_normal, LABEL)
                        mutated_points_normal_list.append(point_mutated_normal)
                    else:
                        mutated_points_normal_list.append("NA")
                    
                    #Appending all the data to the list. Necessary to generate the plots of mutations sequentially.
                    iteration_list.append(iteration)
                    fitness_function_att.append(fitness_mutant_att)
                    prediction_function_att.append(pred_class_mutant_att[0][LABEL])
                    fitness_function_normal.append(fitness_mutant_normal)
                    prediction_function_normal.append(pred_class_mutant_normal[0][LABEL])
                    mutant_digit_att_list.append(mutant_digit_att)
                    mutant_digit_normal_list.append(mutant_digit_normal)
                    xai_images_list.append(xai)
                    list_of_svg_points_list.append(list_of_svg_points)
                    list_of_svg_points_list_2.append(list_of_svg_points_2)
                    pred_input_mutant_att_list.append(pred_input_mutant_att[0])
                    pred_input_mutant_normal_list.append(pred_input_mutant_normal[0])

                    #Checking if the prediction of the mutant digit generated by ATTENTION Method is different from the ground truth (label)
                    if pred_input_mutant_att[0] != LABEL:                        
                        if miss_classification_found_att == False:
                            iterations_detection_att_list.append(iteration)
                            iterations_detection_att = iteration

                            #Writing data to the stats.csv file - Data with the predicitions of both mutated digits (NORMAL and ATTENTION), Label and iteration
                            with open(csv_path, "a") as f1:
                                writer = csv.writer(f1)
                                writer.writerow([image_index, "ATTENTION", METHOD, LABEL, pred_input_mutant_att[0], pred_class_mutant_att[0][LABEL], iteration])   
                        miss_classification_found_att = True
                        if method_winner == None: method_winner = "Heatmaps"

                    #Checking if the prediction of the mutant digit generated by NORMAL Method is different from the ground truth (label)
                    if pred_input_mutant_normal[0] != LABEL:                         
                        if miss_classification_found_normal == False:
                            iterations_detection_normal_list.append(iteration)
                            iterations_detection_normal = iteration
                           
                           #Writing data to the stats.csv file - Data with the predicitions of both mutated digits (NORMAL and ATTENTION), Label and iteration
                            with open(csv_path, "a") as f1:
                                writer = csv.writer(f1)
                                writer.writerow([image_index, "NORMAL", METHOD, LABEL, pred_input_mutant_normal[0], pred_class_mutant_normal[0][LABEL], iteration])                    
                        miss_classification_found_normal = True
                        if method_winner == None: method_winner = "Normal"

                    #If miss classifications were found for both mutation method, we can stop the loop
                    if miss_classification_found_att == True and miss_classification_found_normal == True:
                        break
                    
                    #If Fitness Function calculated for mutated digits are less than the Fitness Function calculated for the previous digit (the digit before the last mutation), 
                    
                    if (fitness_mutant_att < fitness_1) or (fitness_mutant_normal < fitness_2):                        
                        
                        # so we can replace the new "original" digit for the mutated one. ATTENTION Method
                        if (fitness_mutant_att < fitness_1): 
                            if METHOD == "remut":
                                digit_reshaped_1 = mutant_digit_att
                        
                        # so we can replace the new "original" digit for the mutated one. NORMAL Method
                        if (fitness_mutant_normal < fitness_2): 
                            if METHOD == "remut":
                                digit_reshaped_2 = mutant_digit_normal

                #Will save the history of mutated digits only when at least one of method was able to find a missclassification
                # if miss_classification_found_att == True or miss_classification_found_normal == True:

                #If True -> Will save the history of mutated digits indenpendently whether it found a miss classification or not
                if True:
                    if SAVE_IMAGES == True and list_of_svg_points != None:
                        folder_path = create_folder(DST, number_of_mutations, REPETITION, extent, LABEL, image_index, METHOD, "ATT_vs_NOR", run_id) 
                        # save_image(mutant_digit_normal, mutant_digit_att, xai, list_of_svg_points, iteration_list, fitness_function_att, prediction_function_att, fitness_function_normal, prediction_function_normal, number_of_mutations, folder_path, pred_input_mutant_normal[0], pred_input_mutant_att[0], ATTENTION_METHOD, square_size, iteration)
                        save_images(mutant_digit_normal_list, mutant_digit_att_list, xai_images_list, list_of_svg_points_list, iteration_list, fitness_function_att, prediction_function_att, fitness_function_normal, prediction_function_normal, number_of_mutations, folder_path, pred_input_mutant_normal_list, pred_input_mutant_att_list, ATTENTION_METHOD, square_size)
                        make_gif(folder_path, folder_path + "/gif")

                    #Writing data to the stats_2.csv file - Data reagarding a cycle of mutations. 
                    #iterations_detection_att -> The number of iterations ATTENTION method took to find a missclassification 
                    #iterations_detection_normal -> The number of iterations NORAML method took to find a missclassification 
                    #method_winner -> Which method took less iterations to find a missclassification
                    with open(csv_path_2, "a") as f1:
                        writer = csv.writer(f1)
                        writer.writerow([image_index, LABEL, REPETITION, iterations_detection_att, iterations_detection_normal, method_winner])

                    #Writing the points mutated to the .csv 
                    list_to_write = []
                    for iter in range(len(iteration_list)):
                        list_to_write.append([iteration_list[iter], mutated_points_att_list[iter], mutated_points_normal_list[iter], list_of_svg_points_list_2[iter][0]])       

                    with open(csv_path_4, "a") as f1:
                        writer = csv.writer(f1)
                        writer.writerows(list_to_write)
                else:
                    with open(csv_path_2, "a") as f1:
                            writer = csv.writer(f1)
                            writer.writerow([image_index, LABEL, REPETITION, "NA", "NA", "Not Found"])

            #Calculating averages and std dev
            iterations_mean_normal = np.mean(np.array(iterations_detection_normal_list))
            iterations_mean_att = np.mean(np.array(iterations_detection_att_list))
            iterations_std_normal = np.std(np.array(iterations_detection_normal_list))
            iterations_std_att = np.std(np.array(iterations_detection_att_list))

            #Writing data to the stats_3.csv file
            with open(csv_path_3, "a") as f1:
                writer = csv.writer(f1)
                writer.writerow([image_index, LABEL, iterations_mean_att, iterations_mean_normal, iterations_std_att, iterations_std_normal, len(iterations_detection_att_list),len(iterations_detection_normal_list)])
                
    end_time = time.time()
    print("Total Durantion time: ", str(end_time-start_time))
    print("n= ", n)
    print("Number of mutations= ", number_of_mutations)

def option4():
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    model = keras.models.load_model(MODEL)

    n = 100
    images = x_test[:n]
    labels = y_test[:n]
    # print("images.shape", images.shape)

    time_list_seq = []
    time_list_batch = []
    # list_of_n = [100, 250, 500, 750, 1000]
    # list_of_n = [10, 20, 30]
    list_of_n = [i for i in list(range(8001)) if (i % 50 == 0 and i > 0 and i <= 1000) or (i == 2000) or (i == 4000) or (i == 8000)]
    print(list_of_n)

    csv_path = "./xai/stats.csv"
    if os.path.exists(csv_path):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not

    with open(csv_path, append_write) as f1:
        writer = csv.writer(f1)
        writer.writerow(["Number of Images", "Time for Seq", "Time for Batch"])
    for n in list_of_n:

        images = x_test[:n]
        #Batch Method    
        start_time = time.time()
        cams = get_XAI_image(images)
        end_time = time.time()
        delta_time_batch = (end_time - start_time)
        time_list_batch.append(delta_time_batch)
        print("Time to compute heatmaps for " + str(n) + " images in BATCH: ", delta_time_batch)

        #Sequential Method
        start_time = time.time()
        for image_index in range(images.shape[0]):
            cam = get_XAI_image(images[image_index].reshape(1,28,28))
            # plt.imshow(cams[image_index])
            # plt.imshow(images[image_index])
            # plt.savefig("./xai/cam_orig_"+str(image_index)+".jpg")
        end_time = time.time()
        delta_time_seq = (end_time - start_time)
        time_list_seq.append(delta_time_seq)
        print("Time to compute heatmaps for " + str(n) + " images SEQUENTIALLY: ", delta_time_seq)

        with open(csv_path, "a") as f1:
            writer = csv.writer(f1)
            writer.writerow([n, delta_time_seq, delta_time_batch])


    fig, ax = plt.subplots()
    # l_batch, = ax.plot(list_of_n, time_list_batch, "b", label = "Time for Seq")
    # l_seg, = ax.plot(list_of_n, time_list_seq, "r", label = "Time for Batch")
    l_batch = ax.scatter(list_of_n, time_list_batch, c = "blue", label = "Sequentially")
    l_seg = ax.scatter(list_of_n, time_list_seq, c= "red", label = "Batch")
    # ax.legend((l_seg,l_batch), ('Sequentially', "Batch"), loc='upper left', shadow=True)
    ax.legend()
    ax.grid(True)
    ax.set_ylabel('Time (s)')
    ax.set_xlabel('Number of Images (n)')
    ax.set_title('Performance Analysis')
    plt.savefig("./xai/time_analysis.jpg")

if __name__ == "__main__":
    OPTION = 3
    if OPTION == 3:
        option3()




            
    

    




