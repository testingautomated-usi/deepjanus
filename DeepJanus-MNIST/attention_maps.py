import random
import numpy as np

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
from tensorflow.keras.applications.imagenet_utils import preprocess_input

import matplotlib.patches as patches

import time
import rasterization_tools

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

def get_attetion_region(xai_image, orig_image, x_sqr_size, y_sqr_size):
    start_time = time.time()
    x_dim = xai_image.shape[0]
    y_dim = xai_image.shape[1]

    # print("x_dim ",x_dim)
    # print("y_dim ",y_dim)

    greater_value_sum_xai = 0

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

    end_time = time.time()
    return x_final_pos, y_final_pos, (end_time - start_time)

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

def get_attetion_region_mth3(xai_image, svg_path_list, sqr_size, i):
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


    end_time = time.time()
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

    return final_list, (end_time - start_time)


def get_XAI_image(images, model):# images should have the shape: (x, 28, 28) where x>=1

    score = CategoricalScore(0)
    replace2linear = ReplaceToLinear()

    # model = keras.models.load_model(MODEL)    

    # Create GradCAM++ object
    gradcam = GradcamPlusPlus(model,
                            model_modifier=replace2linear,
                            clone=True)

    images_reshaped = input_reshape_images(images)

    X = preprocess_input(images_reshaped, mode = "tf")

    # prediction = model.predict_classes(input_reshape(images))

    # Generate heatmap with GradCAM++
    cam = gradcam(score,
                X,
                penultimate_layer=-1)

    return cam

def getControlPointsInsideAttRegion(x,y,x_dim,y_dim, controlPoints):
    list_of_points = []
    for cp in controlPoints:
        if cp[0] >= (x-1) and cp[0] < x + x_dim + 1:
            if cp[1] >= (y-1) and cp[1] < y + y_dim + 1:
                list_of_points.append(cp)

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

def AM_get_attetion_svg_points_images_mth1(images, x_patch_size, y_patch_size, model):
    """
    AM_get_attetion_svg_points_images_mth1 Iterate all the image looking for the region with more attention and return list of points (tuples) inside the square region with more attention.

    :param images: images should have the shape: (x, 28, 28) where x>=1
    :param x_patch_size: X size of the square region
    :param y_patch_size: Y size of the square region
    :param model: The model object that will predict the value of the digit in the image 
    :return: A list of point positions that are inside the region found. A well detailed explanation about the structure of the list returned is described at the end of this function.
    """ 
    xai = get_XAI_image(images, model)

    # x, y = get_attetion_region(cam, images)
    list_of_ControlPointsInsideRegion = []
    total_elapsed_time = 0
    for i in range(images.shape[0]):
        ControlPoints = vectorization_tools.getImageControlPoints(images[i])
        x, y, elapsed_time = get_attetion_region(xai[i], images[i], x_patch_size, y_patch_size) #Getting coordinates of the highest attetion region (patch) reference point
        total_elapsed_time += elapsed_time
        ControlPointsInsideRegion = getControlPointsInsideAttRegion(x,y,x_patch_size,y_patch_size, ControlPoints) #Getting all the points inside the highest attetion patch
        list_of_ControlPointsInsideRegion.append(ControlPointsInsideRegion)

    return list_of_ControlPointsInsideRegion, total_elapsed_time
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
def AM_get_attetion_svg_points_images_mth2(images, sqr_size, model):
    """
    AM_get_attetion_svg_points_images_mth2 Calculate the attetion score around each SVG path point and return a list of points (tuples) and the respective non-uniform distribution weights for all the SVG path points

    :param images: images should have the shape: (x, 28, 28) where x>=1
    :param sqr_size: X and Y size of the square region
    :param model: The model object that will predict the value of the digit in the image 
    :return: A a list of points (tuples) and the respective non-uniform distribution weights for all the SVG path points. A well detailed explanation about the structure of the list returned is described at the end of this function.
    """ 
    xai = get_XAI_image(images, model)

    # x, y = get_attetion_region(cam, images)
    list_of_points_and_weights = []
    total_elapsed_time = 0
    for i in range(images.shape[0]):
        ControlPoints = vectorization_tools.getImageControlPoints(images[i])
        print("image",i )
        pos_and_prob_list, elapsed_time = get_attetion_region_mth3(xai[i], ControlPoints, sqr_size, i)
        list_of_points_and_weights.append(pos_and_prob_list)
        total_elapsed_time += elapsed_time

    return list_of_points_and_weights, total_elapsed_time
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

#------------Example how to use------------#

#load the MNIST dataset
# mnist = attention_maps.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# model = attention_maps.keras.models.load_model(MODEL)

# images = x_test[:2]

# print("Method1:\n")
# list_of_points_inside_square_attention_patch, elapsed_time = attention_maps.AM_get_attetion_svg_points_images_mth1(images, 3, 3, model)
# print(list_of_points_inside_square_attention_patch,"\n", elapsed_time,"\n")

# print("Method2:\n")
# list_of_points_and_probalities, elapsed_time = attention_maps.AM_get_attetion_svg_points_images_mth2(images, 3, model)
# print(list_of_points_and_probalities,"\n", elapsed_time,"\n")

#------------Pipeline 2 ------------#

# def get_svg_path(image):
#     array = vectorization_tools.preprocess(image)
#     # use Potrace lib to obtain a SVG path from a Bitmap
#     # Create a bitmap from the array
#     bmp = vectorization_tools.potrace.Bitmap(array)
#     # Trace the bitmap to a path
#     path = bmp.trace()
#     return vectorization_tools.createSVGpath(path)

# mnist = keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# model = keras.models.load_model(MODEL)

# n = 10
# images = x_test[:n]
# labels = y_test[:n]
# print("MNIS images shape", images.shape)
# print("Method1:\n")
# for image_index in range(images.shape[0]):
#     print("Image ", str(image_index),"\n")
#     image = images[image_index].reshape(1,28,28)
#     label = labels[image_index]
#     for iteration in range(0,9):
#         print("Iteration ", str(iteration),"\n")
#         list_of_points_inside_square_attention_patch, elapsed_time = AM_get_attetion_svg_points_images_mth1(image, 5, 5, model)
#         # print(list_of_points_inside_square_attention_patch,"\n", elapsed_time,"\n")
#         mutante_digit_path = apply_mutoperator_attention(image, get_svg_path(image[0]), 20, model)
#         # save_svg(vectorization_tools.create_svg_xml(mutante_digit_path),"./mutants/mutant_dig=0_sqr=5_"+str(iteration))
#         rast_nparray = rasterization_tools.rasterize_in_memory(vectorization_tools.create_svg_xml(mutante_digit_path))    
#         # inverted_rast_nparray = 1 - rast_nparray
#         # prediction = np.argmax(model.predict(input_reshape_images(rast_nparray)), axis=-1)
#         prediction = model.predict_classes(input_reshape_images_reverse(rast_nparray))
#         prediction_mnist_data = model.predict_classes(input_reshape_images(image))
#         print("PM: ",str(prediction[0]), " PO:", str(prediction_mnist_data[0]), "Label: ", str(label))        
#         # nparray_to_save = input_reshape_images(inverted_rast_nparray).reshape(28, 28)
#         # plt.imsave("./mutants/Pred_mutant_sqr=5_Pred="+str(prediction[0])+"_lab="+str(labels[0])+"_"+str(iteration)+'.png', nparray_to_save, cmap='gray', format='png')
#         if prediction!= prediction_mnist_data:
#             f, ax = plt.subplots(ncols = 2)
#             ax[0].imshow(rast_nparray.reshape(28, 28), cmap = "gray")
#             ax[0].set_title("Prediction= " + str(prediction[0]))
#             ax[1].imshow(image[0], cmap = "gray")
#             ax[1].set_title("Prediction= " + str(prediction_mnist_data[0]))
#             plt.tight_layout()
#             plt.savefig("./mutants/Pred2_mutant_sqr=5_Pred="+str(prediction[0])+"_PredOrig="+str(prediction_mnist_data[0])+"_lab="+str(label)+"_"+str(iteration)+'.png')





